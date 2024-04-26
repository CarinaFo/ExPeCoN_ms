function [seq,s] = seq(s)
% [seq,s] = seq(s) returns a sequence matrix (seq) that is shuffled 
% per block as well as the random number 
% generator state that was set.
%
% seq: block x trial x condition cue x stimulus type x ITI
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
%   s.ITI
%
% Author:           Carina Forster
% Last update:      18/1/2023

%%

% 
% s.blocks = 5; % blocks in total
% s.cue_cond = [0.25 0.75]; % cued stimulus frequency
% s.cue_cond_n = [60 60]; % repetition of cued sub-blocks per condition
% % make sure the number of trials is evenly dividable by 0.25
% 
% s.fix_t = [2 3]; % inter-trial interval in s
% 
% s.train_blocks = 1;
% s.train_cue_cond_n = [20 20];
%%
% Set random number generator state
s.rng_state = set_rng_state(s);

seq = [];

for block = 1:(s.train_blocks + s.blocks)
    %loop over blocks

    % Create vector with cue-condition order per block
    % creates a list with 60 high and 60 low expectation trials
    % now we create a list with stimulus and noise trials
    % we loop over the conditions and make sure that we have 75 % signal
    % trials for the high expectation condition and vice versa for low exp.
    
    cue_order = [];
    stim_order = [];
    
    for c = 1:length(s.cue_cond)
         if block <= s.train_blocks
            cue_order = [cue_order; repmat(s.cue_cond(c),s.train_cue_cond_n(c),1)];
            stim_order = [stim_order; repmat(1,s.cue_cond(c)*s.train_cue_cond_n(1),1);
                repmat(0,(1-s.cue_cond(c))*s.train_cue_cond_n(1),1),]; 
         else
            cue_order = [cue_order; repmat(s.cue_cond(c),s.cue_cond_n(c),1)];
            stim_order = [stim_order; repmat(1,s.cue_cond(c)*s.cue_cond_n(1),1);
                repmat(0,(1-s.cue_cond(c))*s.cue_cond_n(1),1),]; 
         end
     end 
    
    % now we concatenate both list and save them as columns
    
    shuf_matrix = Shuffle([cue_order, stim_order],2);
    
    % now we shuffle the order of the pairs, making sure that rows are not
    % mixed up
   
    % Create shuffled ITI per block
    
    if block <= s.train_blocks
        seq_ITI = Shuffle(round_dec(linspace(s.iti_t(1),s.iti_t(2),sum(s.train_cue_cond_n)),4))';
        seq_stimdelay = Shuffle(round_dec(linspace(s.stimdelay_t(1),s.stimdelay_t(2),sum(s.train_cue_cond_n)),4))';
        block_list = [repmat(block, sum(s.train_cue_cond_n),1)];
        trial_list = transpose(1: sum(s.train_cue_cond_n));  
    else
        seq_ITI = Shuffle(round_dec(linspace(s.iti_t(1),s.iti_t(2),sum(s.cue_cond_n)),4))';
        seq_stimdelay = Shuffle(round_dec(linspace(s.stimdelay_t(1),s.stimdelay_t(2),sum(s.cue_cond_n)),4))';
        block_list = [repmat(block, sum(s.cue_cond_n),1)];
        trial_list = transpose(1:sum(s.cue_cond_n));
    end
    
    seq_block = [block_list, trial_list, shuf_matrix, seq_ITI, seq_stimdelay];
    
    seq = [seq; seq_block];
    
    % feed in some sanity checks
    
    % Should be 15 (making sure that the amount of signal and noise
    % trials in each condition are valid
    
    testa = length(seq(seq(:,1)==1 & seq(:,3) == 0.75 & seq(:,4)==1));
    testb = length(seq(seq(:,1)==2 & seq(:,3) == 0.75 & seq(:,4)==0));
    testc = length(seq(seq(:,1)==5 & seq(:,3) == 0.25 & seq(:,4)==1));
    
    testlist = [testa, testb, testc];
    
end
