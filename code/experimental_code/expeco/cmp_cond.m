% Hit rate
HR_high(1) = mean(exp_data1.resp1(exp_data1.seq(:,3)==.75 & exp_data1.seq(:,5)==1));
HR_high(2) = mean(exp_data2.resp1(exp_data1.seq(:,3)==.75 & exp_data2.seq(:,5)==1));
HR_high(3) = mean(exp_data3.resp1(exp_data1.seq(:,3)==.75 & exp_data3.seq(:,5)==1));
HR_high(4) = mean(exp_data4.resp1(exp_data1.seq(:,3)==.75 & exp_data4.seq(:,5)==1));

HR_low(1) = mean(exp_data1.resp1(exp_data1.seq(:,3)==.25 & exp_data1.seq(:,5)==1));
HR_low(2) = mean(exp_data2.resp1(exp_data1.seq(:,3)==.25 & exp_data2.seq(:,5)==1));
HR_low(3) = mean(exp_data3.resp1(exp_data1.seq(:,3)==.25 & exp_data3.seq(:,5)==1));
HR_low(4) = mean(exp_data4.resp1(exp_data1.seq(:,3)==.25 & exp_data4.seq(:,5)==1));

% False alarm rate
FAR_high(1) = mean(exp_data1.resp1(exp_data1.seq(:,3)==.75 & exp_data1.seq(:,5)==0));
FAR_high(2) = mean(exp_data2.resp1(exp_data1.seq(:,3)==.75 & exp_data2.seq(:,5)==0));
FAR_high(3) = mean(exp_data3.resp1(exp_data1.seq(:,3)==.75 & exp_data3.seq(:,5)==0));
FAR_high(4) = mean(exp_data4.resp1(exp_data1.seq(:,3)==.75 & exp_data4.seq(:,5)==0));

FAR_low(1) = mean(exp_data1.resp1(exp_data1.seq(:,3)==.25 & exp_data1.seq(:,5)==0));
FAR_low(2) = mean(exp_data2.resp1(exp_data1.seq(:,3)==.25 & exp_data2.seq(:,5)==0));
FAR_low(3) = mean(exp_data3.resp1(exp_data1.seq(:,3)==.25 & exp_data3.seq(:,5)==0));
FAR_low(4) = mean(exp_data4.resp1(exp_data1.seq(:,3)==.25 & exp_data4.seq(:,5)==0));

% Correction: number of hit + 0.5 devided by number of all stimulus trials +1

%HR_high(1) = (sum(exp_data1.resp1(exp_data1.seq(:,3)==.75 & exp_data1.seq(:,5)==1)) + 0.5) / (length(exp_data1.resp1(exp_data1.seq(:,3)==.75 & exp_data1.seq(:,5)==1))+1);

% d' for 1AFC: z(pH) - z(pF)
d_high = normpdf(HR_high) - normpdf(FAR_high)
d_low = normpdf(HR_low) - normpdf(FAR_low)

% c for 1AFC: -(z(pH) + z(pF))/2
c_high = -(normpdf(HR_high) + normpdf(FAR_high))/2
c_low = -(normpdf(HR_low) + normpdf(FAR_low))/2
