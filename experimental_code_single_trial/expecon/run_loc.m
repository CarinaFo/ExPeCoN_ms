function data = run_loc(s,aio_s,trig_s,ID,PF_params)
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
% Author:           Martin Grund
% Last update:      March 9, 2021


%% Intensity input dialog %%
stim_steps = 1;
data.supra = dlg_1intensity(PF_params,s.supra_d,s.supra_name,stim_steps,s.stim_max,s.stim_dec);

try
%% Stimulus

% Generate random interstimulus interval
ISI = Shuffle(round_dec(linspace(s.localizer_isi(1),s.localizer_isi(2),s.localizer_ntrials),4))';


% Generate default waveform vector
[data.stim_wave, data.stim_offset] = rectpulse2(s.pulse_t,1,aio_s.Rate,s.pre_pulse_t,s.wave_t);

% Generate TTL pulse waveform vector
[data.TTL_wave, data.TTL_offset] = rectpulse2(s.TTL_t,s.TTL_V,aio_s.Rate,s.pre_pulse_t,s.wave_t);

% Stimulus trigger
[data.ao_trigger_pre,...
 data.ao_trigger_post,...
 data.ao_error] = deal(zeros(size(s.localizer_ntrials,1),1));

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

% Compute response text location
[data.window(1),data.window(2)] = Screen('WindowSize',window);

  
%% EXPERIMENTAL PROCEDURE %%
% Priority(1) seems to cause DataMissed events for analog input



%% Trial loop


for i = 1:s.localizer_ntrials

        

    
    %%% FIX %%%
    
    % Set font size for symbols
    Screen('TextSize',window,s.cue_size);
    DrawFormattedText(window,s.fix,'center','center',s.txt_color);
    
    Screen('Flip',window);

    %%% STIMULUS %%%
    
    % Select intensity
    
    
    data.intensity(i,1) = data.supra;
    
    
    % Buffer waveform
    stop(aio_s);    
    queueOutputData(aio_s,[data.stim_wave*data.intensity(i,1) data.TTL_wave]);       
    
    % Takes ~150 ms
    startBackground(aio_s);    
    
    % Stimulus delay
    %data.ao_trigger_pre(i,1) = WaitSecs('UntilTime',data.onset_cue{i,1} + s.stim_delay - (s.pre_pulse_t/1000));
    
    % Start analog output (triggers waveform immediately)
    try
        outputSingleScan(trig_s,0)
        outputSingleScan(trig_s,1)
        outputSingleScan(trig_s,0)
    catch lasterr
        disp(['Trial ', num2str(i), ': ', lasterr.message]);
        data.ao_error(i,1) = 1;
        stop(aio_s);
    end   

    data.ao_trigger_post(i,1) = GetSecs;  
    
    
   
    WaitSecs(ISI(i));
    
  
end


%% End procedures

% Close all screens
sca;


%% Error handling

catch lasterr
    sca;
    rethrow(lasterr);
end