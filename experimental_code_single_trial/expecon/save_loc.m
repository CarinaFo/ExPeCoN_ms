function save_loc(p_data,exp_data,s,file_name_end)
% save_exp(p_data,nt_data,nt,file_name_end)) saves the output (exp_data) of 
% the experiment (run_exp), as well as the settings for run_exp (s).
%
% Additionally, it creates a table with the single trial data in each line.
%
% % Input variables %
%   p_data          - output of participant_data
%   exp_data        - output of run_exp
%   s               - output of setup (setting structure)
%   file_name_end   - string that defines end of filename
%
% Author:           Martin Grund
% Last update:      March 11, 2021

% Setup data logging
file_name = [s.file_prefix p_data.ID];

% Create participant data directory
if ~exist(p_data.dir,'dir');
    mkdir('.',p_data.dir);
end

% Compute time intervals
% exp_data = intervals(s,exp_data);

% Save Matlab variables
mat_file_tmp = [p_data.dir file_name '_data_' file_name_end '.mat'];

if exist(mat_file_tmp, 'file')
    disp('MAT-file of experiment exists. Generated random file name to prevent overwritting.')
    save([p_data.dir file_name '_data_' file_name_end '_' num2str(round(sum(100*clock))) '.mat'],'p_data','exp_data','s');
else
    save([p_data.dir file_name '_data_' file_name_end '.mat'],'p_data','exp_data','s');
end