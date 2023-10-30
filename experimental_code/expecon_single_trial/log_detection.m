function log_detection(thr1F_data,exp_data)
% detection_log(thr1F_data,exp_data) displays the applied intensities and 
% their detection rates in the experimental block, as well as the expected
% detection rates based on the estimated psychometric function in thr1F_run.
%
% Input:
%   thr1F_data      - output of threshold assessment thr1F_run
%   exp_data        - output of experimental block (run_exp)
%
% Author:           Martin Grund, Carina Forster
% Last update:      March, 23, 2023

% Display block number
disp(['Block #' num2str(exp_data.seq(1,1))]);

% Calculate detection rates for all intensities in experiment
% Do not consider trials where no button was pressed or where participants pressed the wrong button

nt_detection = count_resp([exp_data.intensity((exp_data.resp1_btn~=0) & (exp_data.resp1_btn ~=5) & ... 
            (exp_data.resp1_btn ~= 8)) exp_data.resp1((exp_data.resp1_btn~=0) & (exp_data.resp1_btn ~=5) & (exp_data.resp1_btn ~= 8))]);
        
    
% Display expected detection rates
arrayfun(@(intensity) disp(['PF(' num2str(intensity) ' mA) = ' num2str(PAL_Quick(thr1F_data.PF_params_PM,intensity))]), nt_detection(:,1));
    
% Display actual detection rates
disp(nt_detection);

% Display correct responses overall (correct rejections + hits)
resp1_corr = ((1-nt_detection(1,4)) + nt_detection(2,4))/2;

disp(['Correct yes/no-responses (CR + hit): ' num2str(resp1_corr) '%'])
disp(['Wrong responded (yes/no) in ' num2str(sum(exp_data.resp1_btn==0 | exp_data.resp1_btn==5 | exp_data.resp1_btn==8)) ' of ' num2str(length(exp_data.resp1_btn)) ' trials.'])