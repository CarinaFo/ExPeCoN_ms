
function apply_pulse_2F(aio_s,trig_s,thr2F)
% apply_pulse(aio_s,thr2F) allows to familiarize participants with
% electrical finger nerve stimulation in advance of automatic threshold
% assessment (thr2F_run). Experimentator can control via commond window 
% input requests at which finger which intensity shall be applied. 
% Stimulation waveform is defined by thr2F_setup_*.
%
% Input:
%   aio_s           - daq acquisition session object
%   trig_s          - trigger session object
%   thr2F           - settings structure (doc thr2F_setup_*)
%
% Author:           Martin Grund
% Last update:      July 22, 2021

%% Prepare output vectors

% Generate default waveform vector
stim_wave = rectpulse2(thr2F.pulse_t,1,aio_s.Rate,thr2F.pre_pulse_t,thr2F.wave_t);

% Generate TTL pulse waveform vector
TTL_wave = rectpulse2(thr2F.TTL_t,thr2F.TTL_V,aio_s.Rate,thr2F.pre_pulse_t,thr2F.wave_t);


%% Stimulation loop

while 1
    finger_index = input('Finger (1 or 2): ');
    
    if isempty(finger_index)
        break
    end
    
    intensity = input('Stimulus intensity (mA): ');
    
    if isempty(intensity)
        break
    end
    
    % Overscripe test intensity with maximum intensity when they exceed stimulus range
    if intensity > max(thr2F.stim_range)
        disp(['Input (' num2str(intensity) ') above maximum intensity (' num2str(max(thr2F.stim_range)) ' mA).'])
        intensity = max(thr2F.stim_range);
        disp(['-> Applied intensity: ' num2str(intensity) ' mA'])
    end
    
    % Zero negative intensities
    if intensity < 0    
        disp(['Input (' num2str(intensity) ') below zero.'])
        intensity = 0;
        disp(['-> Applied intensity: ' num2str(intensity) ' mA'])
    end
    
    tic
    stop(aio_s);
    toc
    
    tic
    switch finger_index
        
        case 1
            queueOutputData(aio_s,[stim_wave*intensity stim_wave*0 TTL_wave stim_wave*0]);
        case 2
            queueOutputData(aio_s,[stim_wave*0 stim_wave*intensity TTL_wave stim_wave*0]);
        otherwise
            break
    end       
    toc
    
    tic
    startBackground(aio_s);
    toc
    
    tic
    outputSingleScan(trig_s,0)
    outputSingleScan(trig_s,1)
    outputSingleScan(trig_s,0)
    toc
    
    WaitSecs(0.5);
    
end