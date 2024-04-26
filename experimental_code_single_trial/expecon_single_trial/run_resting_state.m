function run_resting_state(s)
% data = run_resting_state(s) runs restings state fixation cross on
% screen for specified amount of time
% 
% % Input variables %
%   s               - settings structure (doc setup)
% 
% Author:           Carina Forster
% Last update:      23.03.2023

%% Screen

window = Screen('OpenWindow',0,s.window_color);
HideCursor;

Screen('TextFont',window,s.txt_font);               
             
% Compute response text location
Screen('WindowSize',window);
    
%%% FIX %%%

% Set font size for symbols
Screen('TextSize',window,s.cue_size);
DrawFormattedText(window,s.fix,'center','center',s.txt_color);

Screen('Flip',window);

WaitSecs(s.resting_state_time);
    
sca;

end
