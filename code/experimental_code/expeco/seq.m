function [seq,s] = seq(s)
% [seq,s] = seq(s) returns a sequence matrix (seq) that is shuffled 
% per sub-block, and sub-blocks per block, as well as the random number 
% generator state that was set.
%
%   seq: block x sub-block x cue condition x trial x stimulus type x ITI
%
% The number of stimulus delays is equal the number of trials per block,
% defined as sum(nt.stim_types_num) and lineraly spaced between the minimum
% and maximum of the defined stimulus delay interval (s.stim_delay).
%
% Relevant settings in nt_setup (doc setup):
%   s.rng_state
%   s.blocks
%   s.cue_cond
%   s.cue_cond_n
%   s.cue_block_trials
%   s.ITI
%
% Author:           Martin Grund
% Last update:      March 9, 2021

%%

% s.blocks = 4; % blocks in total
% s.cue_cond = [0.25 0.75]; % cued stimulus frequenzy
% s.cue_cond_n = [6 6]; % repetition of cued sub-blocks per condition 
% s.cue_block_trials = 12; % number of trials per cued sub-blocks
% 
% s.fix_t = [1 2]; % inter-trial interval in s
%
% s.train_blocks = 1;
% s.train_cue_cond_n = [2 2];

%%

% Set random number generator state
s.rng_state = set_rng_state(s);

seq = [];

for block = 1:(s.train_blocks + s.blocks)

    % Create vector with cue-condition order per block
    cue_order = [];

    for i = 1:length(s.cue_cond)
        if block <= s.train_blocks
            cue_order = [cue_order; repmat(s.cue_cond(i),s.train_cue_cond_n(i),1)];
        else
            cue_order = [cue_order; repmat(s.cue_cond(i),s.cue_cond_n(i),1)];
        end
    end

    % Randomize cue order
    cue_order = Shuffle(cue_order);
    
    % Create shuffled ITI per block
    seq_ITI = Shuffle(round_dec(linspace(s.fix_t(1),s.fix_t(2),sum(s.cue_cond_n)*s.cue_block_trials),4))';
    
    % Loop sub-blocks
    for j = 1:length(cue_order)

        % Create vector with trial type order per sub-block
        % 1 = near-threshold; 0 = null
        trial_vector_tmp = [ones(cue_order(j)*s.cue_block_trials,1); ...
                            zeros((1-cue_order(j))*s.cue_block_trials,1)];

        % Randomize trial type order
        trial_vector_tmp = Shuffle(trial_vector_tmp);
        
        % Add trial numbers
        trial_num = ((1:s.cue_block_trials) + (s.cue_block_trials * (j-1)))';
        trial_vector_tmp = [trial_num trial_vector_tmp];

        sub_block_seq_tmp = [repmat([block j cue_order(j)],s.cue_block_trials,1) trial_vector_tmp seq_ITI(trial_num)];
        
        seq = [seq; sub_block_seq_tmp];
    end

end
