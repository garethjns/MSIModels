%% Generate data set
% Set 12 - AV, matched rates, unmatched-dists, constant matched noise,
% fairly easy
% Set 13 - AV, matched rates, unmatched-dists, random matched noise, harder
% Set 14 - AV, matched rates, unmatched-dists, random un-matched noise


%% Set 12 AV
% Matched rates
% Matched rates
% Non-matched distributions
% Variable start buff, unmatched
% Constant, matched noise

clear params

n = 20000;
params.Fs = 400;
fn = ['stimData_AV_s12_', num2str(n), 'x', num2str(params.Fs), '.mat'];

params.dispWarn = 0;
params.seedDebug = 0;
params.startBuff = 50;
params.endBuff = 500;
params.cutOff = 1300;
params.noiseMag = 1000;
params.startBuff = randi([0,120]);
params.nEvents = randi([3,8])*2;
params.eventFreq = 120;
params.noiseMag = randi([0,10])/1000;

close all
example = TemporalStim(params);
example.plot()
drawnow

soundsA = NaN(n, round(params.Fs*params.cutOff/1000), 'double');
eventsA = NaN(n, round(params.Fs*params.cutOff/1000), 'double');
stimsA = cell(n, 1);
ratesA = zeros(n, 1, 'int16');
soundsV = NaN(n, round(params.Fs*params.cutOff/1000), 'double');
eventsV = NaN(n, round(params.Fs*params.cutOff/1000), 'double');
stimsV = cell(n, 1);
ratesV = zeros(n, 1, 'int16');

for s = 1:n
    
    if ~mod(s, 100)
        disp([num2str(s), '/', num2str(n)])
    end
    
    % AV
    rate = randi([3,8])*2;
    params.startBuff = randi([0,120]);
    
    % A
    params.type = 'Aud';
    params.eventType = 'sine';
    ratesA(s,1) = rate;
    
    params.nEvents = rate;
    stimA = TemporalStim(params);
    
    soundsA(s,:) = stimA.sound;
    eventsA(s,:) = stimA.sound2;
    stimsA{s,1} = stimA;
    
    % V
    params.type = 'Vis';
    params.eventType = 'flat';
    ratesV(s,1) = rate;
    
    stimV = TemporalStim(params);
    
    soundsV(s,:) = stimV.sound;
    eventsV(s,:) = stimV.sound2;
    stimsV{s,1} = stimV;
    
    % disp(stim.params.nEvents)
end

save(fn, 'eventsA', 'soundsA', 'ratesA', 'eventsV', 'soundsV', 'ratesV')


%% Set 13 AV
% Matched rates
% Matched rates
% Non-matched distributions
% Variable start buff, unmatched
% Variable, matched noise

clear params

n = 20000;
params.Fs = 400;
fn = ['stimData_AV_s13_', num2str(n), 'x', num2str(params.Fs), '.mat'];

params.dispWarn = 0;
params.seedDebug = 0;
params.startBuff = 50;
params.endBuff = 500;
params.cutOff = 1300;
params.startBuff = randi([0,120]);
params.nEvents = randi([3,8])*2;
params.eventFreq = 120;
params.noiseMag = randi([0,10])/100;

close all
example = TemporalStim(params);
example.plot()
drawnow

soundsA = NaN(n, round(params.Fs*params.cutOff/1000), 'double');
eventsA = NaN(n, round(params.Fs*params.cutOff/1000), 'double');
stimsA = cell(n, 1);
ratesA = zeros(n, 1, 'int16');
soundsV = NaN(n, round(params.Fs*params.cutOff/1000), 'double');
eventsV = NaN(n, round(params.Fs*params.cutOff/1000), 'double');
stimsV = cell(n, 1);
ratesV = zeros(n, 1, 'int16');


for s = 1:n
    
    if ~mod(s, 100)
        disp([num2str(s), '/', num2str(n)])
    end
    
    % AV
    rate = randi([3,8])*2;
    params.startBuff = randi([0,160]);
    params.noiseMag = randi([0,10])/100;
    
    % A
    params.type = 'Aud';
    params.eventType = 'sine';
    ratesA(s,1) = rate;
    params.nEvents = rate;
    
    stimA = TemporalStim(params);
    
    soundsA(s,:) = stimA.sound;
    eventsA(s,:) = stimA.sound2;
    stimsA{s,1} = stimA;
    
    % V
    params.type = 'Vis';
    params.eventType = 'flat';
    ratesV(s,1) = rate;
    
    stimV = TemporalStim(params);
    
    soundsV(s,:) = stimV.sound;
    eventsV(s,:) = stimV.sound2;
    stimsV{s,1} = stimV;
    
    % disp(stim.params.nEvents)
end

save(fn, 'eventsA', 'soundsA', 'ratesA', 'eventsV', 'soundsV', 'ratesV')


%% Set 14 AV
% Matched rates
% Non-matched distributions
% Variable start buff, unmatched
% Variable, unmatched noise

clear params

n = 100000;
params.Fs = 400;
fn = ['stimData_AV_s14_', num2str(n), 'x', num2str(params.Fs), '.mat'];

params.dispWarn = 0;
params.seedDebug = 0;
params.startBuff = 50;
params.endBuff = 500;
params.cutOff = 1300;
params.startBuff = randi([0,120]);
params.nEvents = randi([3,8])*2;
params.eventFreq = 120;
params.noiseMag = randi([0,10])/100;

close all
example = TemporalStim(params);
example.plot()
drawnow

soundsA = NaN(n, round(params.Fs*params.cutOff/1000), 'double');
eventsA = NaN(n, round(params.Fs*params.cutOff/1000), 'double');
stimsA = cell(n, 1);
ratesA = zeros(n, 1, 'int16');
soundsV = NaN(n, round(params.Fs*params.cutOff/1000), 'double');
eventsV = NaN(n, round(params.Fs*params.cutOff/1000), 'double');
stimsV = cell(n, 1);
ratesV = zeros(n, 1, 'int16');


for s = 1:n
    
    if ~mod(s, 100)
        disp([num2str(s), '/', num2str(n)])
    end
    
    % AV
    rate = randi([3,8])*2;
    params.startBuff = randi([0,160]);
    
    % A
    params.type = 'Aud';
    params.eventType = 'sine';
    ratesA(s,1) = rate;
    params.noiseMag = randi([0,10])/100;
    params.nEvents = rate;
    
    stimA = TemporalStim(params);
    
    soundsA(s,:) = stimA.sound;
    eventsA(s,:) = stimA.sound2;
    stimsA{s,1} = stimA;
    
    % V
    params.type = 'Vis';
    params.eventType = 'flat';
    ratesV(s,1) = rate;
    params.noiseMag = randi([0,10])/100;
    
    stimV = TemporalStim(params);
    
    soundsV(s,:) = stimV.sound;
    eventsV(s,:) = stimV.sound2;
    stimsV{s,1} = stimV;
    
    % disp(stim.params.nEvents)
end

save(fn, 'eventsA', 'soundsA', 'ratesA', 'eventsV', 'soundsV', 'ratesV')


%% Set 15 A, V
% A, V,
% Uni only. Off modality is random mag background noise.
% Variable start buff
% Variable noise

clear params

n = 20000;
params.Fs = 400;
fn = ['stimData_AV_s15_', num2str(n), 'x', num2str(params.Fs), '.mat'];

params.dispWarn = 0;
params.seedDebug = 0;
params.startBuff = 50;
params.endBuff = 1300;
params.cutOff = 1300;
params.startBuff = randi([0,120]);
params.nEvents = 0; % randi([3,8])*2;
params.eventFreq = 120;
params.noiseMag = randi([1,10])/100;
params.eventMag = 1;

close all
example = TemporalStim(params);
example.plot()
drawnow

soundsA = NaN(n, round(params.Fs*params.cutOff/1000), 'double');
eventsA = NaN(n, round(params.Fs*params.cutOff/1000), 'double');
stimsA = cell(n, 1);
ratesA = zeros(n, 1, 'int16');
soundsV = NaN(n, round(params.Fs*params.cutOff/1000), 'double');
eventsV = NaN(n, round(params.Fs*params.cutOff/1000), 'double');
stimsV = cell(n, 1);
ratesV = zeros(n, 1, 'int16');
type = NaN(n, 1);

for s = 1:n
    
    if ~mod(s, 100)
        disp([num2str(s), '/', num2str(n)])
    end
    
    % Select a modaility
    m = randi(2); % A or V;
    type(s,1) = m;
    
    % Generate A and V channels, but set off channel to low event mag.
    switch m
        case 1 % A
            aRate = randi([3,8])*2;
            vRate = 0;
        case 2 % V
            aRate = 0;
            vRate = randi([3,8])*2;
    end
    
    % AV
    params.startBuff = randi([0,160]);
    params.endBuff = 1300;
    params.cutOff = 1300;
    params.eventMag = 1;
    
    % A
    params.type = 'Aud';
    params.eventType = 'sine';
    ratesA(s,1) = aRate;
    params.noiseMag = randi([1,6])/100;
    params.nEvents = aRate;
    
    stimA = TemporalStim(params);
    
    soundsA(s,:) = stimA.sound;
    eventsA(s,:) = stimA.sound2;
    stimsA{s,1} = stimA;
    
    % V
    params.type = 'Vis';
    params.eventType = 'flat';
    ratesV(s,1) = vRate;
    params.noiseMag = randi([1,6])/100;
    params.nEvents = vRate;
    
    stimV = TemporalStim(params);
    
    soundsV(s,:) = stimV.sound;
    eventsV(s,:) = stimV.sound2;
    stimsV{s,1} = stimV;
    
    % disp(stim.params.nEvents)
end

save(fn, 'eventsA', 'soundsA', 'ratesA', 'eventsV', 'soundsV', ...
    'ratesV', 'type')


%%  Test

clc
close all

r = randi([1, n]);
disp(['Mod : ', num2str(type(r))])
disp(['aRate : ', num2str(ratesA(r))])
disp(['vRate : ', num2str(ratesV(r))])

stimsV{r}.plot()
stimsA{r}.plot()

figure
subplot(2,1,1)
plot(soundsA(r,:))
subplot(2,1,2)
plot(eventsA(r,:))
suptitle('Aud from vars')

figure
subplot(2,1,1)
plot(soundsV(r,:))
subplot(2,1,2)
plot(eventsV(r,:))
suptitle('Vis from vars')


%% Set 15 A, V
% A, V,
% Uni only. Off modality is random mag background noise.
% Variable start buff
% Variable noise

clear params

n = 20000;
params.Fs = 400;
fn = ['stimData_AV_s16_', num2str(n), 'x', num2str(params.Fs), '.mat'];

params.dispWarn = 0;
params.seedDebug = 0;
params.startBuff = 50;
params.endBuff = 1300;
params.cutOff = 1300;
params.startBuff = randi([0,120]);
params.nEvents = 0; % randi([3,8])*2;
params.eventFreq = 120;
params.noiseMag = randi([1,10])/100;
params.eventMag = 1;

close all
example = TemporalStim(params);
example.plot()
drawnow

soundsA = NaN(n, round(params.Fs*params.cutOff/1000), 'double');
eventsA = NaN(n, round(params.Fs*params.cutOff/1000), 'double');
stimsA = cell(n, 1);
ratesA = zeros(n, 1, 'int16');
soundsV = NaN(n, round(params.Fs*params.cutOff/1000), 'double');
eventsV = NaN(n, round(params.Fs*params.cutOff/1000), 'double');
stimsV = cell(n, 1);
ratesV = zeros(n, 1, 'int16');
type = NaN(n, 1);

for s = 1:n
    
    if ~mod(s, 100)
        disp([num2str(s), '/', num2str(n)])
    end
    
    % Select a modaility
    m = randi(3); % A or V;
    type(s,1) = m;
    
    % Generate A and V channels, but set off channel to low event mag.
    aRate = 0;
    vRate = 0;
    if m==1 || m==3
        aRate = randi([3,8])*2;
    end
    if m==2 || m==3
        vRate = aRate;
    end
    
    % AV
    params.startBuff = randi([0,160]);
    params.endBuff = 1300;
    params.cutOff = 1300;
    params.eventMag = 1;
    
    % A
    params.type = 'Aud';
    params.eventType = 'sine';
    ratesA(s,1) = aRate;
    params.noiseMag = randi([1,6])/100;
    params.nEvents = aRate;
    
    stimA = TemporalStim(params);
    
    soundsA(s,:) = stimA.sound;
    eventsA(s,:) = stimA.sound2;
    stimsA{s,1} = stimA;
    
    % V
    params.type = 'Vis';
    params.eventType = 'flat';
    ratesV(s,1) = vRate;
    params.noiseMag = randi([1,6])/100;
    params.nEvents = vRate;
    
    stimV = TemporalStim(params);
    
    soundsV(s,:) = stimV.sound;
    eventsV(s,:) = stimV.sound2;
    stimsV{s,1} = stimV;
    
    % disp(stim.params.nEvents)
end

save(fn, 'eventsA', 'soundsA', 'ratesA', 'eventsV', 'soundsV', ...
    'ratesV', 'type')


%%  Test

clc
close all

r = randi([1, n]);
disp(['Mod : ', num2str(type(r))])
disp(['aRate : ', num2str(ratesA(r))])
disp(['vRate : ', num2str(ratesV(r))])

stimsV{r}.plot()
stimsA{r}.plot()

figure
subplot(2,1,1)
plot(soundsA(r,:))
subplot(2,1,2)
plot(eventsA(r,:))
suptitle('Aud from vars')

figure
subplot(2,1,1)
plot(soundsV(r,:))
subplot(2,1,2)
plot(eventsV(r,:))
suptitle('Vis from vars')

