import os
import json

def load_refyoutube_json(base_image_dir, is_train=True, set_name=None):
    # 1. 基础路径设置
    base_image_dir = os.path.join(base_image_dir, 'Refer-YouTube-VOS')

    # 划分：train / valid
    data_split = "train" if is_train else "valid"
    if set_name is not None:
        data_split = set_name

    # 图像和 mask 路径
    image_root = os.path.join(base_image_dir, data_split, 'JPEGImages')
    mask_root = os.path.join(base_image_dir, data_split, 'Annotations')

    # ⭐ 关键：meta_expressions.json 的多候选路径
    meta_base = os.path.join(base_image_dir, 'meta_expressions')
    cand1 = os.path.join(meta_base, data_split, 'meta_expressions.json')
    cand2 = os.path.join(meta_base, 'meta_expressions', data_split, 'meta_expressions.json')

    if os.path.isfile(cand1):
        json_file = cand1
    elif os.path.isfile(cand2):
        json_file = cand2
    else:
        raise FileNotFoundError(
            "Ref-Youtube-VOS meta_expressions.json 不存在，尝试过：\n"
            f"  {cand1}\n"
            f"  {cand2}"
        )

    # 2. 加载 JSON 数据
    ann_file = json_file
    with open(str(json_file), 'r') as f:
         subset_expressions_by_video = json.load(f)['videos']
    
    # 【修复缩进】这一行必须和上面的 with 对齐 (4个空格)
    videos = list(subset_expressions_by_video.keys())

    # 3. 验证集过滤逻辑 (增加了文件存在性检查，防止报错)
    if set_name is not None and 'valid' in set_name:
        valid_test_videos = videos
        test_meta_file = json_file.replace('valid/meta_expressions.json', 'test/meta_expressions.json')
        
        # 【新增】检查测试集文件是否存在，存在才过滤，不存在就跳过
        if os.path.exists(test_meta_file):
            try:
                with open(test_meta_file, 'r') as f_test:
                    test_data = json.load(f_test)["videos"]
                test_videos = set(test_data.keys())
                valid_videos = list(set(valid_test_videos) - test_videos)
                # 【关键】更新 videos 列表，让下面的循环使用过滤后的结果
                videos = sorted(valid_videos) 
                print(f"Validation videos filtered: {len(videos)}")
            except Exception as e:
                print(f"Warning: Error filtering validation videos: {e}")
        else:
            print(f"Warning: Test meta file not found at {test_meta_file}. Skipping validation filtering.")

    metas = []
    vid_list = []
    vid2masks = {}

    # 4. 根据模式加载数据
    if data_split == "train":
        for vid in videos:
            # 安全检查：防止过滤后的 key 在字典里找不到
            if vid not in subset_expressions_by_video: continue
            
            vid_data = subset_expressions_by_video[vid]
            vid_frames = sorted(vid_data['frames'])
            vid_len = len(vid_frames)
            if vid_len < 4:
                continue
            vid_list.append(vid)
            vid2masks[vid] = [os.path.join(mask_root, vid, vid_frames[i]+'.png') for i in range(vid_len)]
            for exp_id, exp_dict in vid_data['expressions'].items():
                meta = {}
                meta['str_id'] = f'youtube_{vid}_{exp_id}'
                meta['video'] = vid
                meta['exp'] = exp_dict['exp']
                meta['obj_id'] = [int(exp_dict['obj_id'])]
                meta['anno_id'] = [int(exp_dict['obj_id'])]
                meta['frames'] = vid_frames
                meta['exp_id'] = exp_id
                meta['category'] = None
                meta['length'] = vid_len
                meta['file_names'] = [os.path.join(image_root, vid, vid_frames[i]+ '.jpg') for i in range(vid_len)]
                metas.append(meta)
    else:
        for vid in videos:
            if vid not in subset_expressions_by_video: continue
            
            vid_data = subset_expressions_by_video[vid]
            vid_frames = sorted(vid_data['frames'])
            vid_len = len(vid_frames)
            if vid_len < 4:
                continue
            vid_list.append(vid)
            for exp_id, exp_dict in vid_data['expressions'].items():
                meta = {}
                meta['str_id'] = f'youtube_{vid}_{exp_id}'
                meta['video'] = vid
                meta['exp'] = exp_dict['exp']
                meta['obj_id'] = None
                meta['anno_id'] = None
                meta['frames'] = vid_frames
                meta['exp_id'] = exp_id
                meta['category'] = None
                meta['length'] = vid_len
                meta['file_names'] = [os.path.join(image_root,  vid, vid_frames[i]+ '.jpg') for i in range(vid_len)]
                metas.append(meta)
                
    return vid_list, metas, vid2masks