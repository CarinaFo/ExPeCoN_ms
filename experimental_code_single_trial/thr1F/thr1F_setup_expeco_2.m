
function thr1F = thr1F_setup_respirationCA(stim_range)
% thr1F_setup_*(stim_range) provides an access to the settings
% of the threshold estimation procedure thr1F_run.m
% 
% It creates a structure "thr1F" with all relevant settings that can be
% later on easily modified, e.g. "thr1F.trials_psi = 50;"
%
% % Input %
%   stim_range            - range of possible stimulus intensities
%
%
% % Settings %
%
%   % General
%   thr1F.rng_state         - state of random number generator (not mandatory)
%   thr1F.file_prefix       - prefix for saving files (e.g., analog input recording)
%
%   % Up/down method
%   thr1F.UD_up             - number of misses to increase intensity
%   thr1F.UD_down           - number of hits to decrease intensity
%   thr1F.UD_stepSizeUp     - intensity step up
%   thr1F.UD_stepSizeDown   - intensity step down
%   thr1F.UD_stopCriterion  - 'trials' or 'reversals'
%   thr1F.UD_stopRule       - number of <stopCriterion>
%   thr1F.UD_startValue     - first intensity
%   thr1F.UD_xMax           - maximum intensity
%   thr1F.UD_xMin           - minimum intensity
%
%   thr1F.UD_meanNumber     - number of <stopCriterion> for mean analysis
%
%   For details: doc PAL_AMUD_setupUD
%                doc PAL_AMUD_analyzeUD
%
%
%   % Psi method settings
%   thr1F.trials_psi        - trial number with psi method intensity
%   thr1F.PF                - Psychometric function assumed by Psi method
%	thr1F.stim_range        - considered stimulus intensities
%
%   %% Priors considered for posterior distribution
%   thr1F.priorAlphaRange   - not available, defined by up/down method results
%   thr1F.priorBetaRange    - slope vector
%   thr1F.priorGammaRange   - guess rate vector/scalar
%   thr1F.priorLambdaRange  - lapse rate vector/scalar
%
%   thr1F.priorAlphaSteps   - step size for prior alpha range
%	thr1F.UD_range_factor	  - factors for up/down mean range (min/max) that set prior alpha range
%
%   For details: doc PAL_AMPM_setupPM
%
%
%   % Procedure
%   thr1F.psi_null_rate - ratio of trials w/o stimulation (psi method)
%	thr1F.near_p      - performance level for near-threshold intensity (e.g., 0.40)
%   thr1F.supra_p 	- performance level for supra-threshold intensity (e.g., 0.90)
%	thr1F.trials_test - trial numbers for each "intensity" in thr1F.stim_test
%   thr1F.stim_test   - performance levels for test intensities (> guess & < lapse rate)
%
%
%   % Stimulus
%   thr1F.pulse_t     - duration of stimulus (rectangular pulse) in ms
%   thr1F.pre_pulse_t - duration of before stimulus (rectangular pulse) in ms
%   thr1F.wave_t      - duration of waveform with rectangular pulse in ms
%   thr1F.stim_dec    - decimal position test intensities are rounded to
%   thr1F.TTL_t       - TTL pulse length (1 ms)
%   thr1F.TTL_V       - TTL pulse amplitude (5 V)
%
%
%   % Trial design

%   thr1F.fix_t       - min/max vector for duration of fixation cross in s
%   thr1F.cue_t       - duration of stimulus cue in s
%   thr1F.stim_delay  - min/max vector for pseudo-randomized stimulus delay in s
%   thr1F.resp_window - 'variable'/'fixed' = stop or continue after first button press
%   thr1F.maxRespTime - maximum response time in s
% 
%
%   % Screen design
%   thr1F.window_color- screen background rgb color vector
%
%   %% Text
%   thr1F.txt_color   - text rgb color vector
%   thr1F.txt_font    - font type
%   thr1F.txt_size    - font size for response screen
%   thr1F.cue_size    - font size for fix, cue and pause screens
%   thr1F.cue_color   - cue text rgb color vector
%
%   %% Messages
%   thr1F.fix         - fixation symbol (e.g., '+')
%   thr1F.cue         - cue symbol (e.g., '~+~')
%   thr1F.resp_txt    - response options, yes-no-order important, e.g. ['J'; 'N']
%   thr1F.resp1_offset- 1st response text offset left from screen center 
%
%   % Buttons
%   thr1F.lpt_adr1    - parallel port address (see device manager - LPT1 details - resources)
%   thr1F.lpt_adr2    - as above but 'I/O Range' #2
%   thr1F.lpt_dir     - parallel port direction ('bi' or 'uni')
%   thr1F.btn         - button codes for response options
%   thr1F.btn_esc     - button code for quiting experiment
%
%   %% Instruction
%   thr1F.instr_dir   		  - instruction directory with condition subdirectories
%                     			(relative to directory of thr1F_run.m)
%   thr1F.instr_subdir_wildcard - instruction subdirectory wildcard
%   thr1F.instr_img_wildcard    - filename wildcard of instruction image files
%
%    
% Author:           Martin Grund
% Last update:      December 18, 2018

%% Settings

%% General
% thr1F.rng_state = sum(100*clock); % If not defined, generated by thr1F_run.
thr1F.file_prefix = 'thr1F_';

%% Up/down method settings
thr1F.UD_up = 1;
thr1F.UD_down = 2;
thr1F.UD_stepSizeUp = 0.1;
thr1F.UD_stepSizeDown = 0.1;
thr1F.UD_stopCriterion = 'trials';
thr1F.UD_stopRule = 55; %25
thr1F.UD_startValue = mean(stim_range);
thr1F.UD_xMin = min(stim_range);
thr1F.UD_xMax = max(stim_range);
thr1F.UD_meanNumber = 20;

%% Psi method settings
thr1F.trials_psi = 10;
thr1F.PF = @PAL_Quick;
thr1F.stim_range = stim_range;
thr1F.priorBetaRange = 0.3:.1:1.5;
thr1F.priorGammaRange = .03;
thr1F.priorLambdaRange = .03;

thr1F.priorAlphaSteps = .05;
thr1F.UD_range_factor = [.85; 1.15];

%% Procedure
thr1F.psi_null_rate = 0; % percentage of total trials in psi block -> thr1F.trials_psi ~ (1 - thr1F.null_rate)

thr1F.near_p = .50; %.45;
thr1F.supra_p = .95; % < maximum lapse rate (max(thr1F.priorLambdaRange))

%thr1F.trials_test = [5; 10; 5;];
%thr1F.stim_test = [0; thr1F.near_p; thr1F.supra_p;]; % performance levels > guess & < lapse rate
thr1F.trials_test = [5; 10;];
thr1F.stim_test = [0; thr1F.near_p;]; % performance levels > guess & < lapse rate

%% Stimulus
thr1F.pulse_t = 0.2; % ms
thr1F.pre_pulse_t = 1;
thr1F.wave_t = 3; % ms [thr1F.pre_pulse_t + thr1F.TTL_t + x]
thr1F.stim_dec = 2;

thr1F.TTL_t = 1; % Check if < thr1F.wave_t
thr1F.TTL_V = 5;

%% Trial design
thr1F.fix_t = [0.500 0.500];
thr1F.cue_t = 0.500;
thr1F.stim_delay = [0.4 0.4];
thr1F.resp_window = 'variable';
thr1F.maxRespTime = 2.000;

%% Screen design
thr1F.window_color = [200 200 200]; %[250 250 250]; % grey98 (very light grey)

%% Text
thr1F.txt_color = [40 40 40]; % grey15-16 (anthracite)
thr1F.txt_font = 'Arial';
thr1F.txt_size = 105;
thr1F.cue_size = 125;
%thr1F.cue_color = [255 126 121]; % salmon
thr1F.cue_color = [205 38 38];

%% Messages
thr1F.fix = '+';
thr1F.cue = '+';
thr1F.resp_txt = ['J'; 'N'];
thr1F.resp_offset = 30;

%% Buttons
thr1F.lpt_adr1 = '4000';% '378';
thr1F.lpt_adr2 = '4008';% hex2dec(thr1F.lpt_adr1) + 1024
thr1F.lpt_dir = 'bi';
thr1F.debounceDelay = 0.025;
thr1F.btn = [3 4];
thr1F.btn_esc = 2;
%% Instruction
thr1F.instr_dir = '/instr_expeco2/';
thr1F.instr_subdir_wildcard = 'condition_*';
thr1F.instr_img_wildcard = 'Folie*.png';