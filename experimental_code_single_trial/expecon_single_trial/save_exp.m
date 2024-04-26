function save_exp(p_data,exp_data,s,file_name_end)
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
% Author:           Martin Grund, Carina Forster
% Last update:      23.03.2023

% Setup data logging

file_name = [s.file_prefix p_data.ID];

% Create participant data directory
if ~exist(p_data.dir,'dir');
    mkdir('.',p_data.dir);
end

% Compute time intervals
exp_data = intervals(s,exp_data);

% Save Matlab variables
mat_file_tmp = [p_data.dir file_name '_data_' file_name_end '.mat'];

if exist(mat_file_tmp, 'file')
    disp('MAT-file of experiment exists. Generated random file name to prevent overwritting.')
    save([p_data.dir file_name '_data_' file_name_end '_' num2str(round(sum(100*clock))) '.mat'],'p_data','exp_data','s');
else
    save([p_data.dir file_name '_data_' file_name_end '.mat'],'p_data','exp_data','s');
end


%% Save trial data

% Make nt_data easily accessbile
d = exp_data;

% Get current date
date_str = datestr(now,'yyyy/mm/dd');

% Open file
data_file = fopen([p_data.dir file_name '_trials_' file_name_end '.txt'],'a');

% Write header
fprintf(data_file,'ID\tage\tgender\tdate\tblock\ttrial\tcue\tstim_type\tintensity\tresp1\tresp1_t\tresp1_btn\tresp2\tresp2_t\tresp2_btn\tITI\n');

for i = 1:length(d.seq)
   fprintf(data_file,'%s\t%s\t%s\t%s\t%.0f\t%.0f\t%.2f\t%.0f\t%.2f\t%.0f\t%.4f\t%.0f\t%.0f\t%.4f\t%.0f\t%.3f\n', ...
       p_data.ID,p_data.age,p_data.gender,date_str,d.seq(i,1),d.seq(i,2), ...
       d.seq(i,3),d.seq(i,4),d.intensity(i),d.resp1(i),d.resp1_t(i), ...
       d.resp1_btn(i),d.resp2(i),d.resp2_t(i),d.resp2_btn(i),d.seq(i,5));
end

fclose(data_file);

% NEW: save timings

% Open file
data_file = fopen([p_data.dir file_name '_t_' file_name_end '.txt'],'a');

% Write header
fprintf(data_file,'ID\tdate\tonset_fix_iti\tonset_cue_cond\tonset_fix\tonset_resp1\tonset_resp2\n');

for i = 1:length(d.seq)
   fprintf(data_file,'%s\t%s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n',p_data.ID,date_str,d.onset_fix_iti{i,1},d.onset_cue_cond{i,1},d.onset_fix{i,1},d.onset_resp1{i,1},d.onset_resp2{i,1});
end

fclose(data_file);