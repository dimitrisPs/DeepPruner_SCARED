from pathlib import Path


def datacollector(root_dir, cherry_pick=True):
    root_folder_path = Path(root_dir)

    left_paths = sorted([str(path) for path in (root_folder_path / 'rect_left').iterdir()])
    right_paths = sorted([str(path) for path in (root_folder_path / 'rect_right').iterdir()])
    disp_paths = sorted([str(path) for path in (root_folder_path / 'clean_disparity_128').iterdir()])

    left_train_paths = []
    right_train_paths = []
    disparity_train_paths = []
    
    left_eval_paths = []
    right_eval_paths = []
    disparity_eval_paths = []    

    if cherry_pick:
        good = [0,1,2,4,5,6,8,9,11,12,13,14,31,32,33,34,29]
        eval_samples = [3,7,10,25,26,27,28]
        for ids in good:
            left_train_paths.append(left_paths[ids])
            right_train_paths.append(right_paths[ids])
            disparity_train_paths.append(disp_paths[ids])
        for ids in eval_samples:
            left_eval_paths.append(left_paths[ids])
            right_eval_paths.append(right_paths[ids])
            disparity_eval_paths.append(disp_paths[ids])
    else:
        left_train_paths = left_paths[:15]
        left_train_paths.extend(left_paths[30:35])
        
        right_train_paths = right_paths[:15]
        right_train_paths.extend(right_paths[30:35])
        
        disparity_train_paths = disp_paths[:15]
        disparity_train_paths.extend(disp_paths[30:35])
        
        left_eval_paths = left_paths[25:30]
        right_eval_paths = right_paths[25:30]
        disparity_eval_paths = disp_paths[25:30]
            
    return left_train_paths, right_train_paths, disparity_train_paths, left_eval_paths, right_eval_paths, disparity_eval_paths
