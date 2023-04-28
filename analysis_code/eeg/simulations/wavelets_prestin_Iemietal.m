% This script illustrates the effect of various settings of  time
% frequency analysis using wavelets.

% adapted from Iemi et al. 2017, JoN
% changes made by Carina Forster, 28.04.2023
% length of morlet wavelets to estimate "proper" prestimulus effects

%% Create sine wave

clear all

D = 4;       % total signal duration in seconds.
sigD = 1;    % duration of the test oscillation within the signal.
F = 20;      % frequency of the test oscillationin Hz.
P = .25;     % Phase of the test oscillation. 2 pi radians = 360 degrees
srate = 250; % sampling rate, i.e. N points per sec used to represent sine wave.
T = 1/srate; % sampling period, i.e. for this e.g. points at 1 ms intervals
t = [T:T:D]; % time vector.
myphi=2*pi*P;

sigpoints = length(t)/2 - (sigD*srate)/2:(length(t)/2 + (sigD*srate)/2)-1;
mysig = zeros(1,D*srate);
mysig(sigpoints) = sin(2*F*t(sigpoints)*pi+ myphi);

t_onset = t(sigpoints(1));

%% TF parameters

cycles = 4;
freqscale = 'linear';
nfreqs    = 23; % nr of freqs that will result
freqsout  = linspace(7, 30, nfreqs);
n_cycles    = freqsout/cycles; % constant wavelength depending on frequency

%% TF computation
[tf, outfreqs, outtimes] = timefreq(mysig', srate, ...
    'cycles', n_cycles, 'wletmethod', 'dftfilt3', 'freqscale', freqscale, ...
    'freqs', freqsout);

tfamp = abs(tf./sum(tf(:)));%get the power from the amplitude
% tfamp = abs(tf);%get the power from the amplitude
lwav  = n_cycles.*(1./outfreqs); % wavelet length at each frequency
[wavelet,cycles,freqresol,timeresol] = dftfilt3(freqsout, cycles, srate);

%% TF Plots

% plot 1: wavelength at stimulus onset (timepoint zero)

clf
fh = figure(1);
set(fh, 'color', 'w')

subplot(3,1,1); hold all
[~, waveleti] = min(abs(outfreqs-F));
timeres = timeresol(waveleti);

l = length(real(wavelet{waveleti}));
x = linspace(0,lwav(waveleti),l);
x = x-0.5*lwav(waveleti);
plot(x, real(wavelet{waveleti}))

set(gca, 'xlim', [-0.5 0.5])
set(gca, 'xtick', [ -0.5 0 0.5 1 1.5] )
set(gca, 'FontSize', 20)

set(gca, 'ylim', [-3 3])
set(gca, 'box', 'off')
ylabel('Amplitude');
title(['Wavelet, ' sprintf('%2.1f',n_cycles(waveleti)), ...
    ' cycles at ' num2str(outfreqs(waveleti)) ' Hz; \sigma_t = ' sprintf('%2.2f',timeres) ' sec'])
legend('wavelet')

yl = get(gca, 'ylim');
fill([-timeres, timeres, timeres, -timeres], [yl(1) yl(1) yl(2) yl(2)], 'r', 'edgecolor', 'none', 'facealpha', 0.3)

% plot 2: wavelet amplitude (power) at timepoint zero and prestimulus

subplot(3,1,2); hold all
 plot(t-t_onset,mysig, 'k');
 plot(outtimes./1000-1.5,tfamp(waveleti,:), 'k:');

set(gca, 'xlim', [-0.5 0.5])
set(gca, 'xtick', [ -0.5 0 0.5 1 1.5] )
set(gca, 'FontSize', 20)
 
yl = get(gca, 'ylim');
fill([-timeres, timeres, timeres, -timeres], [yl(1) yl(1) yl(2) yl(2)], 'r', 'edgecolor', 'none', 'facealpha', 0.3)


ylabel('Amplitude');
legend('signal', 'wavelet amplitude')
title([num2str(F) ' Hz signal and wavelet amplitude at ' num2str(F) ' Hz'])

% plot 3: time frequency representation of post stimulus power modulation
% and temporal smearing into prestimulus window

subplot(3,1,3)
hold all
imagesc(outtimes./1000-t_onset,outfreqs,tfamp);
set(gca, 'ylim', [2 30])
set(gca, 'ytick', ([2 10 20 30]) )
set(gca, 'xlim', [-0.5 0.5])
set(gca, 'xtick', [ -0.5 0 0.5] )
set(gca, 'FontSize', 20)

caxis([0 1])
h= colorbar;
ylabel(h, 'Power (\muV^2)', 'FontSize', 25);

daspect([1 100 1])
% axis xy
% colorbar
plot([t(sigpoints(1))   t(sigpoints(1))]-t_onset, [0 30], 'w')
plot([t(sigpoints(end)) t(sigpoints(end))]-t_onset, [0 30], 'w')

% Visualize when it is "safe" to interpret something as "prestimulus"
x1 = t(sigpoints(1)) - lwav/2; % onset of oscillation - half a wavelet length
x2 = t(sigpoints(1)) - timeresol; % onset - timeresol (i.e. 2 * sigma_t sensu Tallon-Baudry)

plot(x2-t_onset, outfreqs, 'r')

xlabel('Time (seconds)', 'FontSize', 25 );
ylabel('Frequency (Hz)', 'FontSize', 25 );
title('Time-frequency representation', 'FontSize', 25 )