function [] = extractOpticalFlow(vid_list)

load_dir = '/home/zhenyang//Workspace/data/UCF11/frames';
save_dir = '/home/zhenyang//Workspace/data/UCF11/features/flow_brox';
vid_list_file = '/home/zhenyang/Workspace/data/UCF11/list_traintest.txt';

folderlist = dir(load_dir);
foldername = {folderlist(:).name};
foldername = setdiff(foldername,{'.','..'});

for i = vid_list
    if ~exist(fullfile(save_dir,foldername{i}),'dir')
        mkdir(fullfile(save_dir,foldername{i}));
    end

    filelist = dir(fullfile(load_dir,foldername{i},'*.jpg'));
    for j = 1:(length(filelist)-1)
        im1 = imread(fullfile(load_dir,foldername{i},sprintf('frame-%06d.jpg',j)));
        im2 = imread(fullfile(load_dir,foldername{i},sprintf('frame-%06d.jpg',j+1)));
        flow_img = compute_flow(im1, im2);

        imwrite(flow_img, fullfile(save_dir,foldername{i},sprintf('flow-%06d.png',j)));
	end
    i
end
end