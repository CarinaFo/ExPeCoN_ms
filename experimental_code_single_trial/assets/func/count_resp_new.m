function x_resp_freq = count_resp(x_resp)
%   count_resp counts the number of 'yes' responses coded as 1.
%
%   The input "x_resp" is a matrix of "intensites x responses", where 
%   respones are either 0 or 1 (e.g., no or yes).
%
%   Ouput matrix:
%   [intensites x frequency of 1 x frequency of intensity x probability of 1]
%   % lay terms: how many times did the participant say yes in catch trials
    % and how many times did the participant say yes in signal trials

%   Author:           Martin Grund, Carina Forster
%   Last update:      23.03.2023

x_resp(:,1) = round_dec(x_resp(:,1),2);
x_unique = unique(x_resp(:,1));
x_resp_freq = zeros(length(x_unique),4);

for i = 1:length(x_unique)
    resp_codes = x_resp(x_resp(:,1)==x_unique(i),2);
    x_resp_freq(i,1:4) = [x_unique(i) sum(resp_codes) numel(resp_codes) mean(resp_codes)];
end