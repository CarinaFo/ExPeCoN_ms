function data = intervals(s,data)
% data = intervals(s,data) computes the actual time intervals between
% screen onsets (e.g., fixation to cue) and analog output trigger in run_exp.
%
% Input:
%   s        - run_exp settings structure (doc setup)
%   data     - run_exp output data structure (e.g., exp_data1)
%   
% Author:           Martin Grund, Carina Forster
% Last update:      February 14 2023

%%

% Trial screen flips
[data.t_fix_cue,...
 data.t_cue_fix,...
 data.t_fix_resp1,...
 data.t_resp1_resp2,...
 data.t_resp2_iti,...
 data.t_cue_ao_trigger_pre_diff_stim_delay,...
 data.trigger_delay_cue,...
 data.trigger_delay_stim,...
 data.trigger_delay_resp,...
 ] = deal(cell(size(data.seq,1),1));

for i = 1:size(data.seq,1)

%% SCREEN ONSET INTERVALS

    data.t_fix_cue{i,1} = (data.onset_fix_iti{i,1}-data.onset_cue_cond{i,1})*1000; % fix iti to cue interval
    data.t_cue_fix{i,1} = (data.onset_fix{i,1}-data.onset_cue_cond{i,1})*1000; % cue to fix interval
    data.t_fix_resp1{i,1} = (data.onset_resp1{i,1}-data.onset_fix{i,1})*1000; % fix to response 1 interval
    data.t_resp1_resp2{i,1} = (data.onset_resp2{i,1} - data.onset_resp1{i,1})*1000; % response 1 to response 2 interval

    if i < size(data.seq,1)
        data.t_resp2_iti{i,1} = (data.onset_fix_iti{i+1,1} - data.onset_resp2{i,1})*1000; % response 2 to fixation iti
    end

%% Trigger timings

    data.t_cue_ao_trigger_pre_diff_stim_delay{i,1} = (data.ao_trigger_pre{i,1}-data.onset_fix{i,1}-s.stim_delay)*1000;
    data.trigger_delay_cue{i,1} = data.onset_cue_cond{i,1} - data.ao_cue_post{i,1};
    data.trigger_delay_stim{i,1} = data.ao_cue_post{i,1} - data.onset_fix{i,1};
    data.trigger_delay_resp{i,1} = data.ao_resp_post{i,1} - (data.onset_resp1{i,1} + data.resp1_t(i,1));

% STIMULUS ONSET LOCKED TO FIRST FIXATION ONSET
data.t_fix_stim_onset{i,1}  = data.ao_trigger_pre{i,1} + data.stim_offset/1000 - data.onset_fix{i,1};

data.t_trigger_ao{i,1}  = data.ao_trigger_post{i,1} - data.ao_trigger_pre{i,1};
end
