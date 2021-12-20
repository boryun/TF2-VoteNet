% this is a all-in-one script of matlab SUNRGBD dataset extract scripts from votenet.
%! Original Author: Charles R. Qi


%% Overall Config
%! absolute path, modify as need
dataset_root = "/media/boyu/depot/DataSet/SUNRGBD";  
output_root = "/home/boyu/桌面/ATLAS/datasets/SUNRGBD_RAW";

%! switchs, modify as need1231
extract_index_file = 1;
extract_v2_data = 1;
extract_v1_label = 1;

if ~exist(output_root, 'dir')
    mkdir(output_root)
end

%% Generate train & val index file.
if extract_index_file
    % load split file
    split = load(strcat(dataset_root, '/SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat'));
    Meta3DBB = load(strcat(dataset_root, '/SUNRGBDMeta3DBB_v2.mat'));

    % Construct Hash Map
    hash_train = java.util.Hashtable;
    hash_val = java.util.Hashtable;

    N_train = length(split.alltrain);
    N_val = length(split.alltest);

    for i = 1:N_train
        folder_path = split.alltrain{i};
        folder_path(1:16) = '';
        hash_train.put(folder_path,0);
    end
    for i = 1:N_val
        folder_path = split.alltest{i};
        folder_path(1:16) = '';
        hash_val.put(folder_path,0);
    end
    
    % Map data to train or val set.
    train_idx_path = strcat(output_root, '/train_data_idx.txt');
    val_idx_path = strcat(output_root, '/val_data_idx.txt');
    fid_train = fopen(train_idx_path, 'w');
    fid_val = fopen(val_idx_path, 'w');

    for imageId = 1:10335
        data = Meta3DBB.SUNRGBDMeta(imageId);
        depthpath = data.depthpath;
        depthpath(1:16) = '';
        [filepath,name,ext] = fileparts(depthpath);
        [filepath,name,ext] = fileparts(filepath);
        if hash_train.containsKey(filepath)
            fprintf(fid_train, '%d\n', imageId);
        elseif hash_val.containsKey(filepath)
            fprintf(fid_val, '%d\n', imageId);
        else
            a = 1;
        end
    end
    fclose(fid_train);
    fclose(fid_val);

    % copy the idx file to meta folder
    script_path = mfilename('fullpath');
    split_idx = strfind(script_path, '/');
    meta_dir = strcat(script_path(1:split_idx(end)-1), '/', 'meta');
    copyfile(train_idx_path, meta_dir);
    copyfile(val_idx_path, meta_dir);
end


%% Extract v2 data (pointcloud, image, calibration matrixs, instance label, semantic label)
if extract_v2_data
    % load meta data
    Meta2DBB = load(strcat(dataset_root, '/SUNRGBDMeta2DBB_v2.mat'));
    if ~(exist("Meta3DBB") == 1)  % avoid duplicate load when "extract_index_file == 1"
        Meta3DBB = load(strcat(dataset_root, '/SUNRGBDMeta3DBB_v2.mat'));
    end

    % create folder
    depth_folder = strcat(output_root, '/depth/');
    image_folder = strcat(output_root, '/image/');
    calib_folder = strcat(output_root, '/calib/');
    det_label_folder = strcat(output_root, '/label/');
    seg_label_folder = strcat(output_root, '/seg_label/');
    mkdir(depth_folder);
    mkdir(image_folder);
    mkdir(calib_folder);
    mkdir(det_label_folder);
    mkdir(seg_label_folder);

    % add path for "read3dPoints" function
    addpath(strcat(dataset_root, '/SUNRGBDtoolbox/readData'))

    % Extrace data
    parfor imageId = 1:10335;
        disp(imageId);
        try
            data = Meta3DBB.SUNRGBDMeta(imageId);
            data.depthpath(1:16) = '';
            data.depthpath = strcat(dataset_root, data.depthpath);
            data.rgbpath(1:16) = '';
            data.rgbpath = strcat(dataset_root, data.rgbpath);
    
            % Write point cloud in depth map
            [rgb,points3d,depthInpaint,imsize]=read3dPoints(data);
            rgb(isnan(points3d(:,1)),:) = [];
            points3d(isnan(points3d(:,1)),:) = [];
            points3d_rgb = [points3d, rgb];
    
    
            % MAT files are 3x smaller than TXT files. In Python can use
            % scipy.io.loadmat('xxx.mat')['points3d_rgb'] to load the data.
            mat_filename = strcat(num2str(imageId,'%06d'), '.mat');
            txt_filename = strcat(num2str(imageId,'%06d'), '.txt');
            parsave(strcat(depth_folder, mat_filename), points3d_rgb);
    
            % Write images
            copyfile(data.rgbpath, sprintf('%s/%06d.jpg', image_folder, imageId));
    
            % Write calibration
            dlmwrite(strcat(calib_folder, txt_filename), data.Rtilt(:)', 'delimiter', ' ');
            dlmwrite(strcat(calib_folder, txt_filename), data.K(:)', 'delimiter', ' ', '-append');
    
            % Write 2D and 3D box label
            data2d = Meta2DBB.SUNRGBDMeta2DBB(imageId);
            fid = fopen(strcat(det_label_folder, txt_filename), 'w');
            for j = 1:length(data.groundtruth3DBB)
                centroid = data.groundtruth3DBB(j).centroid;
                classname = data.groundtruth3DBB(j).classname;
                orientation = data.groundtruth3DBB(j).orientation;
                coeffs = abs(data.groundtruth3DBB(j).coeffs);  %! the coeffs is width(dy), length(dx), height(dz) divide by 2, not the box size!
                box2d = data2d.groundtruth2DBB(j).gtBb2D;
                assert(strcmp(data2d.groundtruth2DBB(j).classname, classname));
                fprintf(fid, '%s %d %d %d %d %f %f %f %f %f %f %f %f\n', classname, box2d(1), box2d(2), box2d(3), box2d(4), centroid(1), centroid(2), centroid(3), coeffs(1), coeffs(2), coeffs(3), orientation(1), orientation(2));
            end
            fclose(fid);
        catch
        end
    end
end


%% extract V1 labels
if extract_v1_label
    % load meta data
    Meta = load(strcat(dataset_root, '/SUNRGBDtoolbox/Metadata/SUNRGBDMeta.mat'));

    % create folder
    det_label_v1_folder = strcat(output_root, '/label_v1/');
    mkdir(det_label_v1_folder);

    for imageId = 1:10335;
        disp(imageId);
        try
            data = Meta.SUNRGBDMeta(imageId);
            data.depthpath(1:16) = '';
            data.depthpath = strcat(dataset_root, '/SUNRGBD', data.depthpath);
            data.rgbpath(1:16) = '';
            data.rgbpath = strcat(dataset_root, '/SUNRGBD', data.rgbpath);
    
            % MAT files are 3x smaller than TXT files. In Python we can use
            % scipy.io.loadmat('xxx.mat')['points3d_rgb'] to load the data.
            mat_filename = strcat(num2str(imageId,'%06d'), '.mat');
            txt_filename = strcat(num2str(imageId,'%06d'), '.txt');
    
            % Write 2D and 3D box label
            data2d = data;
            fid = fopen(strcat(det_label_v1_folder, txt_filename), 'w');
            for j = 1:length(data.groundtruth3DBB)
                centroid = data.groundtruth3DBB(j).centroid;
                classname = data.groundtruth3DBB(j).classname;
                orientation = data.groundtruth3DBB(j).orientation;
                coeffs = abs(data.groundtruth3DBB(j).coeffs);  %! the coeffs is width(dy), length(dx), height(dz) divide by 2, not the box size!
                box2d = data2d.groundtruth2DBB(j).gtBb2D;
                fprintf(fid, '%s %d %d %d %d %f %f %f %f %f %f %f %f\n', classname, box2d(1), box2d(2), box2d(3), box2d(4), centroid(1), centroid(2), centroid(3), coeffs(1), coeffs(2), coeffs(3), orientation(1), orientation(2));
            end
            fclose(fid);
        catch
        end
    end    
end


%% local helper function
function parsave(filename, instance)
    save(filename, 'instance');
end
