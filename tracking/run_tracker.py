import numpy as np
import os
import sys
import time
import argparse
import yaml, json
from PIL import Image
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, '.')
from modules.model import SAMDNet, BCELoss, set_optimizer
from modules.sample_generator import SampleGenerator
from modules.utils import overlap_ratio
from tracking.data_prov import RegionExtractor
from tracking.data_prov import AugRegionExtractor
from tracking.bbreg import BBRegressor
from tracking.gen_config import gen_config
from tracking.edge_detect import max_overlap
from tracking.Grid_search import grid_search
opts = yaml.safe_load(open('E:\SAMDNet\\tracking\options.yaml','r'))


def forward_samples(model, image, samples, out_layer='conv3',preprocessing = False):
    model.eval()
    if preprocessing == False:
        extractor = RegionExtractor(image, samples, opts)
    else:
        extractor = AugRegionExtractor(image, samples, opts)
    for i, regions in enumerate(extractor):
        if opts['use_gpu']:
            regions = regions.cuda()
        with torch.no_grad():
            feat = model(regions, out_layer=out_layer)
        if i==0:
            feats = feat.detach().clone()
        else:
            feats = torch.cat((feats, feat.detach().clone()), 0)
    return feats

def train(model, criterion, optimizer, pos_feats, neg_feats, maxiter, in_layer='fc4'):
    model.train()

    batch_pos = opts['batch_pos']
    batch_neg = opts['batch_neg']
    batch_test = opts['batch_test']
    batch_neg_cand = max(opts['batch_neg_cand'], batch_neg)

    pos_idx = np.random.permutation(pos_feats.size(0))
    neg_idx = np.random.permutation(neg_feats.size(0))
    while(len(pos_idx) < batch_pos * maxiter):
        pos_idx = np.concatenate([pos_idx, np.random.permutation(pos_feats.size(0))])
    while(len(neg_idx) < batch_neg_cand * maxiter):
        neg_idx = np.concatenate([neg_idx, np.random.permutation(neg_feats.size(0))])
    pos_pointer = 0
    neg_pointer = 0

    for i in range(maxiter):

        # select pos idx
        pos_next = pos_pointer + batch_pos
        pos_cur_idx = pos_idx[pos_pointer:pos_next]
        pos_cur_idx = pos_feats.new(pos_cur_idx).long()
        pos_pointer = pos_next

        # select neg idx
        neg_next = neg_pointer + batch_neg_cand
        neg_cur_idx = neg_idx[neg_pointer:neg_next]
        neg_cur_idx = neg_feats.new(neg_cur_idx).long()
        neg_pointer = neg_next

        # create batch
        batch_pos_feats = pos_feats[pos_cur_idx]
        batch_neg_feats = neg_feats[neg_cur_idx]

        # hard negative mining
        if batch_neg_cand > batch_neg:
            model.eval()
            for start in range(0, batch_neg_cand, batch_test):
                end = min(start + batch_test, batch_neg_cand)
                with torch.no_grad():
                    score = model(batch_neg_feats[start:end], in_layer=in_layer)
                if start==0:
                    neg_cand_score = score.detach()[:, 1].clone()
                else:
                    neg_cand_score = torch.cat((neg_cand_score, score.detach()[:, 1].clone()), 0)

            _, top_idx = neg_cand_score.topk(batch_neg)
            batch_neg_feats = batch_neg_feats[top_idx]
            model.train()

        # forward
        pos_score = model(batch_pos_feats, in_layer=in_layer)
        neg_score = model(batch_neg_feats, in_layer=in_layer)

        # optimize
        loss = criterion(pos_score, neg_score)
        model.zero_grad()
        loss.backward()
        if 'grad_clip' in opts:
            torch.nn.utils.clip_grad_norm_(model.parameters(), opts['grad_clip'])
        optimizer.step()


def run_mdnet(img_list, init_bbox, gt=None, savefig_dir='', display=False):

    # Init bbox
    target_bbox = np.array(init_bbox)
    result = np.zeros((len(img_list), 4))
    result_bb = np.zeros((len(img_list), 4))
    result[0] = target_bbox
    result_bb[0] = target_bbox

    if gt is not None:
        overlap = np.zeros(len(img_list))
        overlap[0] = 1

    # Init model
    model = SAMDNet(opts['model_path'])
    if opts['use_gpu']:
        model = model.cuda()

    # Init criterion and optimizer 
    criterion = BCELoss()
    model.set_learnable_params(opts['ft_layers'])
    init_optimizer = set_optimizer(model, opts['lr_init'], opts['lr_mult'])
    update_optimizer = set_optimizer(model, opts['lr_update'], opts['lr_mult'])

    tic = time.time()
    # Load first image
    image = Image.open(img_list[0]).convert('RGB')

    pos_examples = SampleGenerator('gaussian', image.size, opts['trans_pos'], opts['scale_pos'])(
                       target_bbox, opts['n_pos_init'], opts['overlap_pos_init'])
    neg_examples = np.concatenate([
                    SampleGenerator('uniform', image.size, opts['trans_neg_init'], opts['scale_neg_init'])(
                        target_bbox, int(opts['n_neg_init'] * 0.5), opts['overlap_neg_init']),
                    SampleGenerator('whole', image.size)(
                        target_bbox, int(opts['n_neg_init'] * 0.5), opts['overlap_neg_init'])])
    neg_examples = np.random.permutation(neg_examples)
    # Extract pos/neg features
#    pos_feats_aug = forward_samples(model, image, pos_er_examples,preprocessing=True)
#    pos_feats_ori = forward_samples(model, image, pos_examples)
#    pos_feats = np.concatenate((pos_feats_aug.cpu(), pos_feats_ori.cpu()), axis=0)
#   pos_feats = np.random.permutation(pos_feats)
    pos_feats = forward_samples(model, image, pos_examples)
#    pos_feats = torch.from_numpy(pos_feats)
#    pos_feats = pos_feats.cuda()


    neg_feats = forward_samples(model, image, neg_examples)
    # Initial training
    train(model, criterion, init_optimizer, pos_feats, neg_feats, opts['maxiter_init'])
    del init_optimizer, neg_feats
    torch.cuda.empty_cache()

    # Train bbox regressor
    bbreg_examples = SampleGenerator('uniform', image.size, opts['trans_bbreg'], opts['scale_bbreg'], opts['aspect_bbreg'])(
                        target_bbox, opts['n_bbreg'], opts['overlap_bbreg'])
    bbreg_feats = forward_samples(model, image, bbreg_examples)
    bbreg = BBRegressor(image.size)
    bbreg.train(bbreg_feats, bbreg_examples, target_bbox)
    del bbreg_feats
    torch.cuda.empty_cache()

    # Init sample generators for update
    sample_generator = SampleGenerator('gaussian', image.size, opts['trans'], opts['scale'])
    pos_generator = SampleGenerator('gaussian', image.size, opts['trans_pos'], opts['scale_pos'])
    neg_generator = SampleGenerator('uniform', image.size, opts['trans_neg'], opts['scale_neg'])

    # Init pos/neg features for update
    neg_examples = neg_generator(target_bbox, opts['n_neg_update'], opts['overlap_neg_init'])
    neg_feats = forward_samples(model, image, neg_examples)
    pos_feats_all = [pos_feats]
    neg_feats_all = [neg_feats]
    spf_total = time.time() - tic

    # Display
    savefig = savefig_dir != ''
    if display or savefig:
        dpi = 80.0
        figsize = (image.size[0] / dpi, image.size[1] / dpi)

        fig = plt.figure(frameon=False, figsize=figsize, dpi=dpi)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        im = ax.imshow(image, aspect='auto')

        if gt is not None:
            gt_rect = plt.Rectangle(tuple(gt[0, :2]), gt[0, 2], gt[0, 3],
                                    linewidth=3, edgecolor="#00ff00", zorder=1, fill=False)
            ax.add_patch(gt_rect)

        rect = plt.Rectangle(tuple(result_bb[0, :2]), result_bb[0, 2], result_bb[0, 3],
                             linewidth=3, edgecolor="#ff0000", zorder=1, fill=False)
        ax.add_patch(rect)

        if display:
            plt.pause(.01)
            plt.draw()
        if savefig:
            fig.savefig(os.path.join(savefig_dir, '0000.jpg'), dpi=dpi)

    failed_times = 0# Main loop
    index_list = []
    match_target_bbox = np.array([0, 0, 0, 0])
    match_pre_rect = []
    grid_search_rect = []

    for i in range(1, len(img_list)):
        save_pos_feats_flag = True
        tic = time.time()
        # Load image
        image = Image.open(img_list[i]).convert('RGB')

        # Estimate target bbox
        '''
        if failed_times <=2:
            samples = sample_generator(target_bbox, opts['n_samples'])
        else:
            samples = movement_pre(result_bb, opts['n_samples'])
        '''
# draw samples
        dpi = 80.0
        figsize = (image.size[0] / dpi, image.size[1] / dpi)
        fig = plt.figure(frameon=False, figsize=figsize, dpi=dpi)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        im = ax.imshow(image, aspect='auto')

        for s_i in range(len(samples)):
            samples_rect = plt.Rectangle(tuple(samples[s_i, :2]), samples[s_i, 2], samples[s_i, 3], linewidth=1, edgecolor="#00ffff", zorder=1, fill=False)
            ax.add_patch(samples_rect)
        plt.pause(.01)
        plt.draw()
#draw end

        samples = sample_generator(target_bbox, opts['n_samples'])
        sample_scores = forward_samples(model, image, samples, out_layer='fc6')
        top_scores, top_idx = sample_scores[:, 1].topk(5)
        top_idx = top_idx.cpu()
        target_score = top_scores.mean()
        if target_score < 0 and target_bbox.sum() == result_bb[i - 1].sum() and i > 1:
            pass
        else:
            target_bbox = samples[top_idx]
            if top_idx.shape[0] > 1:
                target_bbox = target_bbox.mean(axis=0)
        if target_score < 0:
            failed_times = failed_times + 1
        else:
            failed_times = 0

        target_rect = plt.Rectangle((target_bbox[0], target_bbox[1]), target_bbox[2], target_bbox[3], linewidth=1, edgecolor='#ff0000', zorder=1, fill=False)
        ax.add_patch(target_rect)
        success = target_score > 0
        # Expand search area at failure
        if success:
            sample_generator.set_trans(opts['trans'])
        else:
            sample_generator.expand_trans(opts['trans_limit'])

        # Bbox regression
        if success:
            bbreg_samples = samples[top_idx]
            if top_idx.shape[0] == 1:
                bbreg_samples = bbreg_samples[None, :]
            bbreg_feats = forward_samples(model, image, bbreg_samples)
            bbreg_samples = bbreg.predict(bbreg_feats, bbreg_samples)
            bbreg_bbox = bbreg_samples.mean(axis=0)
        else:
            bbreg_bbox = target_bbox
        # Save result
        '''
        refined_bbox = max_overlap(img_list[i], bbreg_bbox, result_bb[i-1], failed_times)
        if bbreg_bbox[2] * bbreg_bbox[3] > refined_bbox[2] * refined_bbox[3] :
            result_bb[i] = bbreg_bbox
        else:
            samples_refine_bbox = np.tile(refined_bbox[None, :], (256, 1))
            samples_refine_bbox_scores = forward_samples(model, image, samples_refine_bbox, out_layer='fc6')
            refine_top_scores, top_idx = samples_refine_bbox_scores[:, 1].topk(5)
            refine_bbox_score = refine_top_scores.mean()

            samples_bbr_bbox = np.tile(bbreg_bbox[None, :], (256, 1))
            samples_bbr_bbox_scores = forward_samples(model, image, samples_bbr_bbox, out_layer='fc6')
            bbr_top_scores, top_idx = samples_bbr_bbox_scores[:, 1].topk(5)
            bbr_bbox_score = bbr_top_scores.mean()
            
            result_bb[i] = 0.2 * refined_bbox + 0.8 * bbreg_bbox
        '''
        result_bb[i] = bbreg_bbox
        result_bb_rect = plt.Rectangle((result_bb[i][0], result_bb[i][1]), result_bb[i][2], result_bb[i][3], linewidth=1,
                                       edgecolor='#000000', zorder=1, fill=False)
        ax.add_patch(result_bb_rect)
        if failed_times >= 2 and i >= 20:

            new_boxes = grid_search(image_path_current=img_list[i], previous_bbox=result_bb[i])
            new_boxes_scores = forward_samples(model, image, new_boxes, out_layer='fc6')
            new_top_scores, new_top_idx = new_boxes_scores[:, 1].topk(2)
#                target_bbox = result_bb[i-1]
            if new_top_scores[0] > 0 and new_top_scores[1] < 0:
                new_pre_bbox = new_boxes[new_top_idx[0]]
                target_bbox = new_pre_bbox
                new_pre_bbox_rect = plt.Rectangle((new_pre_bbox[0], new_pre_bbox[1]), new_pre_bbox[2], new_pre_bbox[3],
                                            linewidth=1, edgecolor='#00ffff', zorder=1, fill=False)
                ax.add_patch(new_pre_bbox_rect)

                pos_examples = pos_generator(target_bbox, opts['n_pos_update'], opts['overlap_pos_update'])
                pos_feats = forward_samples(model, image, pos_examples)
                pos_feats_all.append(pos_feats)

                nframes = min(opts['n_frames_short'], len(pos_feats_all))
                pos_data = torch.cat(pos_feats_all[-nframes:], 0)
                neg_data = torch.cat(neg_feats_all, 0)
                train(model, criterion, update_optimizer, pos_data, neg_data, opts['maxiter_update'])
            elif new_top_scores[0] > 0 and new_top_scores[0] >= 0:
                new_top_idx = new_top_idx.cpu()
                new_pre_bbox = new_boxes[new_top_idx].mean(axis=0)
                target_bbox = new_pre_bbox
                grid_search_rect.append(plt.Rectangle((new_pre_bbox[0], new_pre_bbox[1]), new_pre_bbox[2],
                                                    new_pre_bbox[3], linewidth=2, edgecolor='#0000ff', zorder=1,
                                                    fill=False))
                if len(grid_search_rect) != 0:
                    ax.add_patch(grid_search_rect[-1])

                pos_examples = pos_generator(target_bbox, opts['n_pos_update'], opts['overlap_pos_update'])
                pos_feats = forward_samples(model, image, pos_examples)
                pos_feats_all.append(pos_feats)

                nframes = min(opts['n_frames_short'], len(pos_feats_all))
                pos_data = torch.cat(pos_feats_all[-nframes:], 0)
                neg_data = torch.cat(neg_feats_all, 0)
                train(model, criterion, update_optimizer, pos_data, neg_data, opts['maxiter_update'])
            elif new_top_scores[0] < 0:
                edge_detect_bbox = max_overlap(image_path=img_list[i], target_bbox=target_bbox, results_history=result_bb[i])
                if edge_detect_bbox.sum() != target_bbox.sum():
                    pos_examples = pos_generator(target_bbox, opts['n_pos_update'], opts['overlap_pos_update'])
                    pos_feats = forward_samples(model, image, pos_examples)
                    pos_feats_all.append(pos_feats)

                    nframes = min(opts['n_frames_short'], len(pos_feats_all))
                    pos_data = torch.cat(pos_feats_all[-nframes:], 0)
                    neg_data = torch.cat(neg_feats_all, 0)
                    train(model, criterion, update_optimizer, pos_data, neg_data, opts['maxiter_update'])
                    target_bbox = edge_detect_bbox
                    edge_detect_rect = plt.Rectangle((edge_detect_bbox[0], edge_detect_bbox[1]), edge_detect_bbox[2], edge_detect_bbox[3],
                                                linewidth=1, edgecolor='#ff00ff', zorder=1, fill=False)
                    ax.add_patch(edge_detect_rect)
                else:
                    save_pos_feats_flag = False
                    new_pre_bbox = result_bb[i]
                    target_bbox = new_pre_bbox
                    new_pre_bbox_rect = plt.Rectangle((new_pre_bbox[0], new_pre_bbox[1]), new_pre_bbox[2], new_pre_bbox[3],
                                                linewidth=1, edgecolor='#ffff00', zorder=1, fill=False)
                    ax.add_patch(new_pre_bbox_rect)

            else:
                pass

            success = 1
            failed_times = 0

        gt_rect = plt.Rectangle(tuple(gt[i, :2]), gt[i, 2], gt[i, 3],
                                linewidth=1, edgecolor="#00ff00", zorder=1, fill=False)
        ax.add_patch(gt_rect)
        fig.savefig(os.path.join(savefig_dir, '{:04d}.jpg'.format(i)), dpi=dpi)
        plt.close()
        # Data collect
        torch.cuda.empty_cache()
        if success and save_pos_feats_flag == True :
            pos_examples = pos_generator(target_bbox, opts['n_pos_update'], opts['overlap_pos_update'])
            pos_feats = forward_samples(model, image, pos_examples)
            pos_feats_all.append(pos_feats)
            if len(pos_feats_all) > opts['n_frames_long']:
                del pos_feats_all[0]
            neg_examples = neg_generator(target_bbox, opts['n_neg_update'], opts['overlap_neg_update'])
            neg_feats = forward_samples(model, image, neg_examples)
            neg_feats_all.append(neg_feats)
            if len(neg_feats_all) > opts['n_frames_short']:
                del neg_feats_all[0]

        # Short term update
        if not success:
            nframes = min(opts['n_frames_short'], len(pos_feats_all))
            pos_data = torch.cat(pos_feats_all[-nframes:], 0)
            neg_data = torch.cat(neg_feats_all, 0)
            train(model, criterion, update_optimizer, pos_data, neg_data, opts['maxiter_update'])
        # Long term update
        elif i % opts['long_interval'] == 0:
            pos_data = torch.cat(pos_feats_all, 0)
            neg_data = torch.cat(neg_feats_all, 0)
            train(model, criterion, update_optimizer, pos_data, neg_data, opts['maxiter_update'])

        torch.cuda.empty_cache()
        spf = time.time() - tic
        spf_total += spf

        # Display
        if display or savefig:
            im.set_data(image)

            if gt is not None:
                gt_rect.set_xy(gt[i, :2])
                gt_rect.set_width(gt[i, 2])
                gt_rect.set_height(gt[i, 3])

            rect.set_xy(result_bb[i, :2])
            rect.set_width(result_bb[i, 2])
            rect.set_height(result_bb[i, 3])

            if display:
                plt.pause(.01)
                plt.draw()
            if savefig:
                a = 1
                #fig.savefig(os.path.join(savefig_dir, '{:04d}.jpg'.format(i)), dpi=dpi)
                #fig.savefig(os.path.join(savefig_dir, '{:04d}.jpg'.format(i)), dpi=dpi)
        if gt is None:
            print('Frame {:d}/{:d}, Score {:.3f}, Time {:.3f}'
                .format(i, len(img_list), target_score, spf))
        else:
            overlap[i] = overlap_ratio(gt[i], result_bb[i])[0]
            print('Frame {:d}/{:d}, Overlap {:.3f}, Score {:.3f}, Time {:.3f}'
                .format(i, len(img_list), overlap[i], target_score, spf))

    if gt is not None:
        print('meanIOU: {:.3f}'.format(overlap.mean()))
    fps = len(img_list) / spf_total
    overlap_ave = overlap.mean()
    return result, result_bb, fps, overlap_ave


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seq', default='', help='input seq')
    parser.add_argument('-j', '--json', default='', help='input json')
    parser.add_argument('-f', '--savefig',default= '',help='input dir')
    parser.add_argument('-d', '--display', action='store_true')

    args = parser.parse_args()
    assert args.seq != '' or args.json != ''

    np.random.seed(0)
    torch.manual_seed(0)

    # Generate sequence config
    img_list, init_bbox, gt, savefig_dir, display, result_path = gen_config(args)

    # Run tracker
    result, result_bb, fps, overlap = run_mdnet(img_list, init_bbox, gt=gt, savefig_dir=savefig_dir, display=display)

    # Save result
    res = {}
    res['res'] = result_bb.round().tolist()
    res['type'] = 'rect'
    res['fps'] = fps
    json.dump(res, open(result_path, 'w'), indent=2)

    #templecolor [20:]
    #OTB2015 [11:]
    #UAV123 [40:]
    #vot2015[43]
    '''
    print(args.seq)
    file_path_bbox = 'E:/SAMDNet/results/' + args.seq[37:] + '.txt'
    fp1 = open(file_path_bbox, 'w')
    data = result_bb.round()
    box = ''
    for i in range(len(data)):
          box = box + str(data[i]) + '\n'

    box = box.replace('[', '')
    box = box.replace(']', '')
    box = box.replace(' ', '')
    box = box.replace('.', ',')
    box = box.strip()
    fp1.write(box)
    fp1.close()

    file_path_fps = 'E:/SAMDNet/results/' + args.seq[37:] + '_fps.txt'
    fp2 = open(file_path_fps,'w')
    fp2.write(str(fps))
    fp2.close()

    file_path_overlap = 'E:/py-MDNet-master-attention/results/' + args.seq[37:] + '_overlap.txt'
    fp3 = open(file_path_overlap,'w')
    fp3.write(str(overlap))
    fp3.close()
    '''