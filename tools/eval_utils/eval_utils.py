import copy
import pickle
import time
import os
import numpy as np
import torch
import tqdm

from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils
from pcdet import device

def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)



    metric['gt_num'] += ret_dict.get('gt', 0)
    # metric['pred_num'][0] +=ret_dict['pred'][0] if ret_dict.get('pred',False) else 0
    # metric['pred_num'][1] +=ret_dict['pred'][1] if ret_dict.get('pred',False) else 0
    # metric['pred_scores'] = torch.concat(metric['pred_scores'],)
    # metric['loss_cls'][0] +=ret_dict['loss_cls'][0] if ret_dict.get('loss_cls',False) else 0.
    # metric['loss_cls'][1] +=ret_dict['loss_cls'][1] if ret_dict.get('loss_cls',False) else 0.

    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST

    disp_dict['recall_%s_%s_%s ' % (str(min_thresh),str(thresh[1]),str(thresh[2]))] = \
            '(%d,%d) (%d,%d) (%d,%d ) / %d /(cls %.8s )' % (
                metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)],
                metric['recall_roi_%s' % str(thresh[1])], metric['recall_rcnn_%s' % str(thresh[1])],
                metric['recall_roi_%s' % str(thresh[2])], metric['recall_rcnn_%s' % str(thresh[2])],
                metric['gt_num'],
                # metric['loss_cls'][0]/metric['pred_num'][0] ,metric['loss_cls'][1]/metric['pred_num'][1],
            torch.var(metric['pred_scores']).item())

def _create_pd_detection(detections, infos, result_path, tracking=False):
    """Creates a prediction objects file."""
    from waymo_open_dataset import label_pb2
    from waymo_open_dataset.protos import metrics_pb2
    LABEL_TO_TYPE = {0:1,1:2,2:4}
    objects = metrics_pb2.Objects()

    for idx,detection in enumerate(detections):
        info = infos[idx]
        # obj = get_obj(info['anno_path'])

        box3d = detection["boxes_lidar"][:,:7]
        scores = detection["score"]
        labels = detection["pred_labels"]-1

        # transform back to Waymo coordinate
        # x,y,z,w,l,h,r2
        # x,y,z,l,w,h,r1
        # r2 = -pi/2 - r1
        box3d[:, -1] = -box3d[:, -1] - np.pi / 2
        box3d = box3d[:, [0, 1, 2, 4, 3, 5, -1]]

        if tracking:
            tracking_ids = detection['tracking_ids']

        for i in range(box3d.shape[0]):
            det  = box3d[i]
            score = scores[i]

            label = labels[i]

            o = metrics_pb2.Object()
            o.context_name = info['metadata']['context_name']
            o.frame_timestamp_micros = info['metadata']['timestamp_micros']

            # Populating box and score.
            box = label_pb2.Label.Box()
            box.center_x = det[0]
            box.center_y = det[1]
            box.center_z = det[2]
            box.length = det[3]
            box.width = det[4]
            box.height = det[5]
            box.heading = det[-1]
            o.object.box.CopyFrom(box)
            o.score = score
            # Use correct type.
            o.object.type = LABEL_TO_TYPE[label]

            if tracking:
                o.object.id = uuid_gen.get_uuid(int(tracking_ids[i]))

            objects.objects.append(o)

    # Write objects to a file.
    if tracking:
        path = os.path.join(result_path, 'tracking_pred.bin')
    else:
        path = os.path.join(result_path, 'detection_pred.bin')

    print("results saved to {}".format(path))
    f = open(path, 'wb')
    f.write(objects.SerializeToString())
    f.close()


def eval_one_epoch(cfg, args, model, dataloader, epoch_id, logger, dist_test=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if args.save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
        'pred_scores':torch.tensor([],device=device),

    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []
    import GPUtil
    if GPUtil.getGPUs()[0].name.endswith('3080'):
        delay_time = 0.2
    else:
        delay_time = 0
    if getattr(args, 'infer_time', False):
        start_iter = int(len(dataloader) * 0.1)
        infer_time_meter = common_utils.AverageMeter()

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            broadcast_buffers=False
        )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)

        if getattr(args, 'infer_time', False):
            start_time = time.time()

        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)

        disp_dict = {}

        if getattr(args, 'infer_time', False):
            inference_time = time.time() - start_time
            infer_time_meter.update(inference_time * 1000)
            # use ms to measure inference time
            disp_dict['infer_time'] = f'{infer_time_meter.val:.2f}({infer_time_meter.avg:.2f})'
        time.sleep(delay_time)
        metric['pred_scores'] = torch.concat([metric['pred_scores'],pred_dicts[0]['pred_scores']],dim=-1)
        statistics_info(cfg, ret_dict, metric, disp_dict)
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if args.save_to_file else None
        )
        det_annos += annos
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()

    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))
    logger.info('Average loss cls : %.8f  ' % (torch.var(metric['pred_scores']).item()))
    if getattr(args, 'infer_time', False):
        logger.info('Average infer time %.4f/frame'%(infer_time_meter.avg))
    if args.output_pkl:
        with open(result_dir / 'result.pkl', 'wb') as f:
            pickle.dump(det_annos, f)

    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )

    logger.info(result_str)
    ret_dict.update(result_dict)

    _create_pd_detection(det_annos,dataset.infos,final_output_dir)


    logger.info('Result is saved to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict


if __name__ == '__main__':
    pass
