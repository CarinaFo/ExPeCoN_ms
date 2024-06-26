%% Expeco
%  Forced-choice experiment with near-threshold somatosensory stimulation
%  - yes/no decision
%  - confidence about yes/no
%  - likelihood of stimulation is cued beforehand
% 
% Author:           Martin Grund

% Last update:      March 9, 2021

% updates by Carina Forster, September 2021
% addeded an additional block (now its 5 experimental blocks)
% a resting state block before the actual task

%% [SKIP IF RE-RUN] Initialize experiment

cd('C:\Users\willi\code\d5_control');

% Experiment directory, threshold assessment directory, number of analog 
% output channels, samples per second
[p_data,aio_s,trig_s,ao] = exp_init_NI_new('expecon','thr1F',2,100000);

%% EXPERIMENT
%% [SKIP IF RE-RUN] Settings for experiment

s = setup;

% Since parallel port addresses change (check in device manager)

%s.lpt_adr1 = '4FF8';
%s.lpt_adr2 = '4FF4';
s.lpt_adr1 = '4000';
s.lpt_adr2 = '4008';

%% Test parallel port (button box)
lpt = dio_setup(s.lpt_adr1,s.lpt_adr2,s.lpt_dir);

[button,respTime,port] = parallel_button(10,GetSecs,'variable',s.debounceDelay,lpt)

clear lpt button respTime port

%% [SKIP IF RE-RUN] Create sequence and save settings

[exp_seq,s] = seq(s);

save([p_data.dir s.file_prefix 'settings_seq.mat'],'p_data','exp_seq','s');

%% resting state block (this shows a fixation cross on the screen while recording EEG)

%after fixing electrodes on finger

run_resting_state(s);


%% Familiarize with electrical finger nerve stimulation
% & coarse threshold assessment to find start value for real threshold
% assessment

thr1F_test = thr1F_setup_expeco(0:.02:4);

apply_pulse_1F_new(aio_s,trig_s,thr1F_test)

% Stops if intensity input is empty

clear thr1F_test

%% ThA 1
block = 1;

thr1F = thr1F_setup_expeco(0:.02:5);
thr1F.lpt_adr1 = s.lpt_adr1;
thr1F.lpt_adr2 = s.lpt_adr2;

%adjust the start Value for each participant based on familiarization

%dialog input for start Value
prompt = {'Enter a start value for threshold assessment: '};
dlgtitle = 'start value input';
dims = [1 35];
startValue = inputdlg(prompt, dlgtitle, dims);
thr1F.UD_startValue = str2num(startValue{1});

%thr1F.UD_startValue = 1.6;

%no need to change things here

% Initial ThA with more trials and coarser steps
thr1F.UD_stopRule = 50;
thr1F.UD_meanNumber = 20;
thr1F.trials_psi = 45;
%thr1F.UD_stepSizeUp = 0.2; # leave step size 0.1
thr1F.UD_stepSizeDown = thr1F.UD_stepSizeUp;

% For test purposes
% thr1F.UD_stopRule = 6;
% thr1F.UD_meanNumber = 2;
% thr1F.trials_psi = 4;
% thr1F.trials_test = [2; 2;];

% thr1F.UD_stopRule = 30;
% thr1F.UD_meanNumber = 15;
% thr1F.UD_startValue = 3.6;
% thr1F.trials_psi = 15;
% thr1F.UD_stepSizeUp = 0.1;
% thr1F.UD_stepSizeDown = thr1F.UD_stepSizeUp;

thr1F_data1 = thr1F_loop(thr1F,p_data,aio_s,trig_s,block,1);

% If repetition necessary:
% (1) Narrow stimulus range
% thr1F = thr1F_setup_expeco(0:.02:2.5);
% (2) Use last threshold estimate as start value
% thr1F.UD_startValue = thr1F_data1.near;
% (3) Indicate another run with last input - thr1F_loop(...,block,run)
% thr1F_data1 = thr1F_loop(thr1F,p_data,aio_s,trig_s,block,2);

%% Train BLOCK == Block 1 (so in total we have 6 blocks saved)

block = 1;

% skip ThA 1 for test purposes: thr1F_data1.PF_params_PM = ones(1,4);

train_data1 = run_exp(s,aio_s,trig_s,p_data.ID,thr1F_data1.PF_params_PM,exp_seq(exp_seq(:,1)==block,:));

train_data1 = intervals(s,train_data1);

save_exp(p_data,train_data1,s,['0' num2str(block)]);

log_detection(thr1F_data1,train_data1);

%% BLOCK 1 - FIRST EXPERIMENTAL BLOCK
block = 2;

exp_data1 = run_exp(s,aio_s,trig_s,p_data.ID,thr1F_data1.PF_params_PM,exp_seq(exp_seq(:,1)==block,:));

exp_data1 = intervals(s,exp_data1);

save_exp(p_data,exp_data1,s,['0' num2str(block)]);

log_detection(thr1F_data1,exp_data1);

%%
%% ThA 2
block = 2;

thr1F = thr1F_setup_expeco(0:.02:4);
thr1F.lpt_adr1 = s.lpt_adr1;
thr1F.lpt_adr2 = s.lpt_adr2;

thr1F.UD_stepSizeUp = 0.05;
thr1F.UD_stepSizeDown = thr1F.UD_stepSizeUp;

thr1F.UD_startValue = exp_data1.near(end,1);

thr1F_data2 = thr1F_loop(thr1F,p_data,aio_s,trig_s,block,1);

% thr1F.UD_startValue = thr1F_data2.near;
% thr1F_data2 = thr1F_loop(thr1F,p_data,aio_s,trig_s,block,2);


%% BLOCK 2
block = 3;

exp_data2 = run_exp(s,aio_s,trig_s,p_data.ID,thr1F_data2.PF_params_PM,exp_seq(exp_seq(:,1)==block,:));

exp_data2 = intervals(s,exp_data2);

save_exp(p_data,exp_data2,s,['0' num2str(block)]);

log_detection(thr1F_data2,exp_data2);

%%
%% ThA 3
block = 3;

thr1F.UD_startValue = exp_data2.near(end,1);
thr1F.UD_stepSizeUp = 0.03;
thr1F.UD_stepSizeDown = thr1F.UD_stepSizeUp;

thr1F_data3 = thr1F_loop(thr1F,p_data,aio_s,trig_s,block,1);
% thr1F.UD_startValue = thr1F_data3.near; %nt_data2.near(end,1);
% thr1F_data3 = thr1F_loop(thr1F,p_data,aio_s,trig_s,block,2);
% thr1F_data3 = thr1F_loop(thr1F,p_data,aio_s,trig_s,block,3);

%% BLOCK 3
block = 4;
% "Fake" ThA: 
% thr1F_data3.PF_params_PM = ones(4,1);

exp_data3 = run_exp(s,aio_s,trig_s,p_data.ID,thr1F_data3.PF_params_PM,exp_seq(exp_seq(:,1)==block,:));

exp_data3 = intervals(s,exp_data3);

save_exp(p_data,exp_data3,s,['0' num2str(block)]);

log_detection(thr1F_data3,exp_data3);

%%
%% ThA 4
block = 4;

thr1F.UD_startValue = exp_data3.near(end,1);
% thr1F.UD_stepSizeUp = 0.2;
% thr1F.UD_stepSizeDown = thr1F.UD_stepSizeUp;

thr1F_data4 = thr1F_loop(thr1F,p_data,aio_s,trig_s,block,1);
% thr1F.UD_startValue = thr1F_data4.near;
% thr1F_data4 = thr1F_loop(thr1F,p_data,aio_s,trig_s,block,2);


%% BLOCK 4
block = 5;

% If ThA #4 is skipped:
%thr1F_data4 = thr1F_data3;

exp_data4 = run_exp(s,aio_s,trig_s,p_data.ID,thr1F_data4.PF_params_PM,exp_seq(exp_seq(:,1)==block,:));

exp_data4 = intervals(s,exp_data4);

save_exp(p_data,exp_data4,s,['0' num2str(block)]);

log_detection(thr1F_data4,exp_data4);

%% BLOCK 5
block = 6;

% If ThA #4 is skipped:
%thr1F_data4 = thr1F_data3;

exp_data5 = run_exp(s,aio_s,trig_s,p_data.ID,thr1F_data4.PF_params_PM,exp_seq(exp_seq(:,1)==block,:));

exp_data5 = intervals(s,exp_data5);

save_exp(p_data,exp_data5,s,['0' num2str(block)]);

log_detection(thr1F_data4,exp_data5);


%% Re-run experiment

% (0) Go to working directory
%
%     cd('C:\Users\willi\code\d5_control');
%
% (1) Load sequence data "expeco_settings_seq.mat" [exp_seq; p_data; s;]
%
% (2) Load last threshold assessment data, e.g. "thr1F_01_data_02_01.mat" [p_data; thr1F; thr1F_data;]
%
% (3) Rename "thr1F_data" to last successful threshold assessment:
%
%     thr1F_data1 = thr1F_data; clear thr1F_data; % If 1st block failed
%     thr1F_data2 = thr1F_data; clear thr1F_data; % If 2nd block failed
%     thr1F_data3 = thr1F_data; clear thr1F_data; % If 3rd block failed
%     thr1F_data4 = thr1F_data; clear thr1F_data; % If 4th block failed
%
% (4) Load last block data, e.g. "expeco_01_data_02.mat" [exp_data; p_data; s;]
%
% (5) Rename 'exp_data' to the loaded block:
%
%     exp_data1 = exp_data; clear exp_data;
%     exp_data2 = exp_data; clear exp_data;
%     exp_data3 = exp_data; clear exp_data;
%     exp_data4 = exp_data; clear exp_data;
%
% (6) Initialize experiment
%
%     cd('C:\Users\willi\code\d5_control')
%
%     [aio_s,trig_s,ao] = exp_re_init_NI_new('expeco','thr1F',p_data,2,100000);
%
% (7) Run "Familiarize with electrical finger nerve stimulation" section (Strg + Enter)
%
% (8) Run "Test parallel port" section
%
% (9) Run block or threshold assessment you want to re-run

%% localizer


% skip ThA 1 for test purposes: thr1F_data1.PF_params_PM = ones(1,4);

loc_data = run_loc(s,aio_s,trig_s,p_data.ID,thr1F_data1.PF_params_PM);

%loc_data = intervals(s,loc_data);

%save_exp(p_data,loc_data,s,['0' num2str(block)]);

%log_detection(thr1F_data1,loc_data);

%% CLOSE

diary off
clear all