function data = run_resting_state(s)
% data = run_resting_state(s) runs restings state block
% 
% % Input variables %
%   s               - settings structure (doc setup)
%
% % Output variables %
%   data            - data output structure
% 
% Author:           Carina Forster, Martin Grund
% Last update:      November 11, 2021


%% Screen

window = Screen('OpenWindow',0,s.window_color);
HideCursor;

Screen('TextFont',window,s.txt_font);               
             
% Compute response text location
[data.window(1),data.window(2)] = Screen('WindowSize',window);
    
%%% FIX %%%

% Set font size for symbols
Screen('TextSize',window,s.cue_size);
DrawFormattedText(window,s.fix,'center','center',s.txt_color);

Screen('Flip',window);

WaitSecs(s.resting_state_time);
    
sca;

end
