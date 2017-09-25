%% Generate data set

%% Set 11 AV
% Matched rates
% Matched rates
% Non-matched distributions
% Variable start buff, unmatched

clear params

n = 10000;
params.Fs = 400;
fn = ['stimData_AV_s11_', num2str(n), 'x', num2str(params.Fs), '.mat'];

params.dispWarn = 0;
params.seedDebug = 0;
params.startBuff = 50;
params.endBuff = 500;
params.cutOff = 1300;
params.noiseMag = 0.0001;

params.eventFreq = 120;

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
    params.startBuff = randi([10,50]);
    
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


%% Set 12 AV
% Matched rates
% Matched rates
% Non-matched distributions
% Variable start buff, unmatched
% Variable noise

clear params

n = 20000;
params.Fs = 400;
fn = ['stimData_AV_s12_', num2str(n), 'x', num2str(params.Fs), '.mat'];

params.dispWarn = 0;
params.seedDebug = 0;
params.startBuff = 50;
params.endBuff = 500;
params.cutOff = 1300;
params.noiseMag = 0.0001;
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