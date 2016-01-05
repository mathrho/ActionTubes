function [] = extractOpticalFlow(vid_list, fps)

load_dir = '/home/zhenyang//Workspace/data/UCF101/frames';
%save_dir = sprintf('/home/zhenyang//Workspace/data/UCF11/features/flow_brox@fps%d', fps);
save_dir = sprintf('/home/zhenyang//Workspace/data/UCF101/features/flow_brox');
%vid_list_file = '/home/zhenyang/Workspace/data/UCF11/list_traintest.txt';
vid_list_file = '/home/zhenyang/Workspace/data/UCF101/list_UCF101.txt';

folderlist = dir(load_dir);
foldername = {folderlist(:).name};
foldername = setdiff(foldername,{'.','..'});

skip = int32(30.0/fps)

for i = vid_list
    if ~exist(fullfile(save_dir,foldername{i}),'dir')
        mkdir(fullfile(save_dir,foldername{i}));
    end

    filelist = dir(fullfile(load_dir,foldername{i},'*.jpg'));
    for j = 1:(length(filelist)-skip)
        im1 = imread(fullfile(load_dir,foldername{i},sprintf('frame-%06d.jpg',j)));
        im2 = imread(fullfile(load_dir,foldername{i},sprintf('frame-%06d.jpg',j+skip)));
        flow_img = compute_flow(im1, im2);

        imwrite(flow_img, fullfile(save_dir,foldername{i},sprintf('flow-%06d.png',j)));
	end
    disp(num2str(i));
end
end