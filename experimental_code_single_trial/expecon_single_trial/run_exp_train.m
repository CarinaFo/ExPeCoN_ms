function data = run_exp(s,aio_s,trig_s,IDS,PF_params,seq)
% run_exp(s,ao,ai,ID,PF_params,seq) starts the experiment.
% 
% % Input variables %
%   s               - settings structure (doc setup)
%   aio_s           - daq acquisition session object
%   trig_s          - trigger session object
%   ID              - participant ID as string (e.g., p_data.ID)
%   PF_params       - psychometric function parameters (e.g., psi_data.PF_params_PM)
% 
% % Output variables %
%   data            - data output structure
% 
% Author:           Martin Grund, Carina Forster
% Last update:      14.02.2023


%% Intensity input dialog %%
stim_steps = 1;
data.near = dlg_1intensity(PF_params,s.near_d,s.near_name,stim_steps,s.stim_max,s.stim_dec);

%% SETUP %%    
try


%% Random number generator
data.rng_state = set_rng_state(s);


%% Response-button mapping

% don't randomize response buttons

data.resp_btn_map = 2;

% 1st response
%switch data.resp_btn_map
    %case {1,2} % JN
data.resp1_txt = s.resp1_txt;
    %case {3,4} % NJ
       % data.resp1_txt = flipud(s.resp1_txt);
%end

% 2nd response
%switch data.resp_btn_map
    %case {1,3} % SU
data.resp2_txt = s.resp2_txt;
    %case {2,4} % US
        %data.resp2_txt = flipud(s.resp2_txt);
%end

%% Sequence (doc seq)

data.seq = seq;


%% Stimulus

% Generate default waveform vector
[data.stim_wave, data.stim_offset] = rectpulse2(s.pulse_t,1,aio_s.Rate,s.pre_pulse_t,s.wave_t);

% Generate TTL pulse waveform vector
[data.TTL_wave, data.TTL_offset] = rectpulse2(s.TTL_t,s.TTL_V,aio_s.Rate,s.pre_pulse_t,s.wave_t);


%% Timing

% how long is one trial (maximum time?) % without feedback
% mean iti = 1.5
%s.prob_t = 1
%s.fix_t = 1
% s.resp1_mean = 0.7
% s.resp1_mean = 0.3

data.trial_t =  mean(s.iti_t) + s.prob_t + s.fix_t + s.resp1_max_t + s.resp2_max_t;

% Trial screen flips
[data.onset_cue_cond,...
 data.onset_fix,...
 data.onset_resp1,...
 data.onset_resp1_p,...
 data.onset_resp2,...
 data.onset_fix_iti,...
 data.onset_feedback] = deal(cell(size(data.seq,1),5));

% % End of wait for trial end
% data.wait_trial_end = zeros(size(data.seq,1),1);

% Before block screen flips
[data.btn_instr,...
data.onset_instr] = deal(cell(1,5));

% Stimulus trigger
[data.ao_trigger_pre,...
 data.ao_trigger_post,...
 data.ao_cue_post,...
 data.ao_resp_post,...
 data.ao_error] = deal(cell(size(data.seq,1),1));

% Response tracking
[data.resp1_btn,...
data.resp1_t,...
data.resp1,...
data.resp2_btn,...
data.resp2_t,...
data.resp2] = deal(zeros(size(data.seq,1),1));

%% Parallel port
lpt = dio_setup(s.lpt_adr1,s.lpt_adr2,s.lpt_dir);

%% Screen

window = Screen('OpenWindow',0,s.window_color);
HideCursor;

Screen('TextFont',window,s.txt_font);               
             
% Get screen frame rate
Priority(1); % recommended by Mario Kleiner's Perfomance & Timing How-to
flip_t = Screen('GetFlipInterval',window,s.get_flip_i);
Priority(0);
data.flip_t = flip_t;

% Update sequence - make fix_t divisible by flip time
data.seq(:,5) = data.seq(:,5) - mod(data.seq(:,5), flip_t);

% Compute response text location
[data.window(1),data.window(2)] = Screen('WindowSize',window);

% left top right bottom (vertical)
% 
resp_rect1 = [data.window(1)*0.5 - s.txt_size,...
              data.window(2)*0.5 - s.resp1_offset - s.txt_size,...
              data.window(1)*0.5 + s.txt_size,...
              data.window(2)*0.5 - s.resp1_offset];
resp_rect2 = [data.window(1)*0.5 - s.txt_size,...
              data.window(2)*0.5 + s.resp1_offset,...
              data.window(1)*0.5 + s.txt_size,...
              data.window(2)*0.5 + s.resp1_offset + s.txt_size];

% horizontal display (options next to each other)

% resp_rect1 = [data.window(1)*0.5 - s.resp1_offset - s.txt_size,...
%               data.window(2)*0.5 - s.txt_size,...
%               data.window(1)*0.5 - s.resp1_offset,...
%               data.window(2)*0.5 + s.txt_size];
% resp_rect2 = [data.window(1)*0.5 + s.resp1_offset,...
%               data.window(2)*0.5 - s.txt_size,...
%               data.window(1)*0.5 + s.resp1_offset + s.txt_size,...
%               data.window(2)*0.5 + s.txt_size];
% 
% resp_rect3 = [data.window(1)*0.5 - 8*s.resp1_offset - s.txt_size,...
%               data.window(2)*0.5 - 8*s.txt_size,...
%               data.window(1)*0.5 - 8*s.resp1_offset,...
%               data.window(2)*0.5 + 8*s.txt_size];
%           
% resp_rect4 = [data.window(1)*0.5 + 8*s.resp1_offset,...
%               data.window(2)*0.5 - 8*s.txt_size,...
%               data.window(1)*0.5 + 8*s.resp1_offset + s.txt_size,...
%               data.window(2)*0.5 + 8*s.txt_size];
 
%% EXPERIMENTAL PROCEDURE %%
% Priority(1) seems to cause DataMissed events for analog input

%% Instructions

% Get directory (based on response-button mapping)
instr_dir = [fileparts(mfilename('fullpath')) s.instr_dir];
instr_subdir = dir([instr_dir s.instr_subdir_wildcard num2str(data.resp_btn_map) '*']);

% Load image data
instr_images = load_images([instr_dir instr_subdir.name '/'],s.instr_img_wildcard);

% Show images
[data.btn_instr, data.onset_instr] = show_instr_img(instr_images,window,lpt);

% Delete image data from memory
clear instr_images img_texture

%% Trial loop

for i = 1:size(data.seq,1)

    %1. show fixation cross on the screen for inter-trial interval time

    % only for first trial
    if i == 1 
        Screen('TextSize',window,s.cue_size);
        DrawFormattedText(window,s.fix,'center','center',s.txt_color);
        [data.onset_fix_iti{i,:}] = Screen('Flip',window);
    % check if feedback on last trial
    elseif (mod((i-1) ,s.feedback_after_trials) == 0)  && (s.show_feedback == 1)
        % ITI dependent on feeback screen
        Screen('TextSize',window,s.cue_size);
        DrawFormattedText(window,s.fix,'center','center',s.txt_color);
        [data.onset_fix_iti{i,:}] = Screen('Flip',window,data.onset_feedback{i-1,1} + s.feedback_t - flip_t);
    % all other trials
    else
        % ITI dependent on confidence response
        Screen('TextSize',window,s.cue_size);
        DrawFormattedText(window,s.fix,'center','center',s.txt_color);
        [data.onset_fix_iti{i,:}] = Screen('Flip',window,data.onset_resp2{i-1,1} + data.resp2_t(i-1,1) - flip_t);
    end
        
    %%% Stimulus probability for every trial %%% Cue trigger == S1
    
    % Buffer waveform and send data to the DAQ card
    
    stop(aio_s);    
    
    queueOutputData(aio_s,[data.stim_wave*0 data.stim_wave*0 data.stim_wave*0 data.TTL_wave]);       
    
    % Takes ~150 ms
    
    startBackground(aio_s);  
    
    % 2. display the probability cue on the screen
    
    Screen('TextSize',window,s.cue_size);
    
    if s.randomise_cues == 1 % if condition == 1 the blue circle shows
        % low expectation trials and vice versa
        if data.seq(i,3) == 0.25 % blue
            DrawFormattedText(window, 'O', 'center','center', s.blue_circle);
        else % orange
            DrawFormattedText(window, 'O', 'center','center', s.orange_circle);
        end
    else
        if data.seq(i,3) == 0.25 % orange
            DrawFormattedText(window, 'O', 'center','center', s.orange_circle);
        else % orange
            DrawFormattedText(window, 'O', 'center','center', s.blue_circle);
        end
    end

    [data.onset_cue_cond{i,:}] = Screen('Flip',window, data.onset_fix_iti{i,1} + data.seq(i,5) - flip_t);
    
   % Start analog output (triggers waveform immediately, mean 20ms)

    try
        outputSingleScan(trig_s,0)
        outputSingleScan(trig_s,1)
        outputSingleScan(trig_s,0)
    catch lasterr
        disp(['Trial ', num2str(i), ': ', lasterr.message]);
        data.ao_error{i,1} = 1;
        stop(aio_s);
    end   
    
    % save compare data.onset_cue_cond{1,1} - data.ao_cue_post(i,1)  %
    % shoudl be small < 20ms

    data.ao_cue_post{i,1} = GetSecs;

    %%% Stimulation interveral: Now show fixation cross on the screen %%%
    
    % Set font size for symbols

    Screen('TextSize',window,s.cue_size);
    DrawFormattedText(window,s.fix,'center','center',s.txt_color);

    % 3. show fixation cross

    [data.onset_fix{i,:}] = Screen('Flip',window,data.onset_cue_cond{i,1} + s.prob_t - flip_t);

    %%% STIMULUS %%% EEG trigger == S2
    
    % Select stimulus intensity
    
    switch data.seq(i,4)
        case 0 % null
            data.intensity(i,1) = 0;
        case 1 % near
            data.intensity(i,1) = data.near;
    end
 
    % Buffer waveform

    stop(aio_s);    

    queueOutputData(aio_s,[data.stim_wave*data.intensity(i,1) data.stim_wave*0 data.TTL_wave  data.stim_wave*0]);       
    
    % Takes ~150 ms

    startBackground(aio_s);    
    
    % Stimulus delay

    data.ao_trigger_pre{i,1} = WaitSecs('UntilTime',data.onset_fix{i,1} + s.stim_delay - (s.pre_pulse_t/1000));
    
    % Start analog output (triggers waveform immediately)

    try
        outputSingleScan(trig_s,0)
        outputSingleScan(trig_s,1)
        outputSingleScan(trig_s,0)
    catch lasterr
        disp(['Trial ', num2str(i), ': ', lasterr.message]);
        data.ao_error{i,1} = 1;
        stop(aio_s);
    end   

    data.ao_trigger_post{i,1} = GetSecs;  

    % data.ao_trigger_post(1,1) - data.ao_trigger_pre(1,1)

    % Buffer waveform for response trigger, Response trigger  in EEG = S3

    stop(aio_s);

    queueOutputData(aio_s,[data.stim_wave*0 data.stim_wave*0 data.TTL_wave data.TTL_wave]);       
    
    % Takes ~150 ms

    startBackground(aio_s);   
    
    %%% RESPONSE 1 - DETECTION %%%
    
    % Response options

    Screen('TextSize',window,s.txt_size);

    DrawFormattedText(window,data.resp1_txt(1,:),'center','center',s.txt_color,[],[],[],[],[],resp_rect1);
    DrawFormattedText(window,data.resp1_txt(2,:),'center','center',s.txt_color,[],[],[],[],[],resp_rect2);
    
    % 4. show detection response options on the screen 
    
   [data.onset_resp1{i,:}] = Screen('Flip',window,data.onset_fix{i,1} + s.fix_t - flip_t);
        
    % Wait for key press

    [data.resp1_btn(i,1),data.resp1_t(i,1),data.resp1_port(i,:)] = parallel_button(s.resp1_max_t,data.onset_resp1{i,1},s.resp_window,s.debounceDelay,lpt);        
     
    % Start analog output (triggers waveform immediately)

    try
        outputSingleScan(trig_s,0)
        outputSingleScan(trig_s,1)
        outputSingleScan(trig_s,0)
    catch lasterr
        disp(['Trial ', num2str(i), ': ', lasterr.message]);
        data.ao_error{i,1} = 1;
        stop(aio_s);
    end
    
    data.ao_resp_post{i,1} = GetSecs;

    % data.ao_resp_post(1,1) - data.onset_resp1{1,1}

    % 5. show fixation cross for s.resp_p_min_t on the screen

%     if s.resp1_max_t-data.resp1_t(i,1) > s.resp_p_min_t
       % Screen('TextSize',window,s.cue_size);
        %DrawFormattedText(window,s.fix,'center','center',s.txt_color);
        %[data.onset_resp1_p{i,:}] = Screen('Flip',window);
%     end        
    
    
    %%% RESPONSE 2 - CONFIDENCE %%%
    
    % Response options

    Screen('TextSize',window,s.txt_size);
%     DrawFormattedText(window,num2str(data.resp2_txt(1,:)),'center','center',s.txt_color,[],[],[],[],[],resp_rect3);
    DrawFormattedText(window,num2str(data.resp2_txt(1,:)),'center','center',s.txt_color,[],[],[],[],[],resp_rect1);
    DrawFormattedText(window,num2str(data.resp2_txt(2,:)),'center','center',s.txt_color,[],[],[],[],[],resp_rect2);
%     DrawFormattedText(window,num2str(data.resp2_txt(4,:)),'center','center',s.txt_color,[],[],[],[],[],resp_rect4);
    
    % 6. show confidence response options on the screen 

    [data.onset_resp2{i,:}] = Screen('Flip',window,data.onset_resp1_p{i,1} + s.resp_p_min_t - flip_t);
      
    % Wait for key press

    [data.resp2_btn(i,1),data.resp2_t(i,1),data.resp2_port(i,:)] = parallel_button(s.resp2_max_t,data.onset_resp2{i,1},s.resp_window,s.debounceDelay,lpt);       
    
    %%% RESPONSE EVALUATION %%%
    
    % Response 1

    switch data.resp1_btn(i,1)
        case num2cell(s.btn_resp1)
            switch data.resp1_txt(s.btn_resp1==data.resp1_btn(i,1))
                case s.resp1_txt(1)
                    data.resp1(i,1) = 1; % yes
                case s.resp1_txt(2)
                    data.resp1(i,1) = 0; % no
            end
        case s.btn_esc
            break
        otherwise
            data.resp1(i,1) = 0;
    end             

    % Response 2
    switch data.resp2_btn(i,1)
        case num2cell(s.btn_resp2)
            switch data.resp2_txt(s.btn_resp2==data.resp2_btn(i,1))
                case s.resp2_txt(1)
                    data.resp2(i,1) = 0; % guessing
                case s.resp2_txt(2)
                    data.resp2(i,1) = 1; % sure
%                 case s.resp2_txt(3)
%                     data.resp2(i,1) = 3; % somewhat confident
%                 case s.resp2_txt(4)
%                     data.resp2(i,1) = 4; % highly confident
            end
        case s.btn_esc
            break
        otherwise
            data.resp2(i,1) = 0;
    end
    
    %%% FEEDBACK %%%
    
    if mod(i,s.feedback_after_trials_train) == 0  && s.show_feedback == 1       
        
        % Calculate detection rates for all intensities in experiment
        % Do not consider trails when no button was pressed [exclusion could be extended]
        %nt_detection = count_resp([data.intensity(data.resp1_btn~=0) data.resp1(data.resp1_btn~=0)]);
        % Do not consider trials where the wrong button was pressed,
        % 06.04.2023, CF
        nt_detection = count_resp([data.intensity((data.resp1_btn~=0) & (data.resp1_btn ~=5) & ... 
            (data.resp1_btn ~= 8)) data.resp1((data.resp1_btn~=0) & (data.resp1_btn ~=5) & (data.resp1_btn ~= 8))]);
        % Display correct responses overall (correct rejections + hits) up
        % to this trial
        resp1_corr = ((1-nt_detection(1,4)) + nt_detection(2,4))/2;

        Screen('TextSize',window,s.cue_size);
        DrawFormattedText(window,[num2str(round(resp1_corr*100)) ...
            '% richtig'],'center','center',s.txt_color);
        % 6. show feedback on the screen if a feedback trial
        [data.onset_feedback{i,:}] = Screen('Flip',window);
        WaitSecs(s.feedback_t);
    end    
    
    %%% WAIT UNTIL TRIAL END %%%

    if i == size(data.seq,1)
        [data.onset_fix{i+1,:}] = Screen('Flip',window);
        data.wait_block_end = WaitSecs('UntilTime',data.onset_resp2{i,1} + s.resp2_max_t);
    end
  
    
end


%% End procedures

% Close all screens
sca;


%% Error handling

catch lasterr
    sca;
    rethrow(lasterr);
end