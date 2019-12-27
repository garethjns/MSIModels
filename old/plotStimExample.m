%% Plot example
% Plots random A and V stim

close all

% Load set 12
a = load('Data\stimData_AV_s12_20000x400.mat');

figure
subplot(2,1,1)
idx = randi(size(a.soundsA, 1));
plot(a.soundsA(idx,:))
hold on
plot(a.eventsA(idx,:))
title(['"Auditory" stim: ', num2str(a.ratesA(idx)), ' events'])
ylabel('Mag.')
legend({'Raw signal', 'Event indicator'})
subplot(2,1,2)
idx = randi(size(a.soundsA, 1));
plot(a.soundsV(idx,:))
hold on
plot(a.eventsV(idx,:))
title(['"Visual" stim: ', num2str(a.ratesV(idx)), ' events'])
xlabel('Time (pts)')
ylabel('Mag.')

ng
