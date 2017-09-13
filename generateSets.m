%% Generate data set

%% Set 2

fn = 'stimData_s2_500x1178.mat';
clear params

n = 500;
params.eventType = 'sine';
params.eventFreq = 220;
params.dispWarn = 0;
params.seedDebug = 0;
params.startBuff = 50;
params.endBuff = 400;
params.cutOff = 1150;
params.Fs = 1024;
params.noiseMag = 0.0001;

sounds = NaN(n, round(params.Fs*params.cutOff/1000), 'double');
events = NaN(n, round(params.Fs*params.cutOff/1000), 'double');
stims = cell(n, 1);
rates = zeros(n, 1, 'int16');

for s = 1:n
    
    if ~mod(s, 100)
        disp([num2str(s), '/', num2str(n)])
    end
    
    rate = randi([6,16]);
    rates(s,1) = rate;
    
    params.nEvents = rate;
    stim = TemporalStim(params);
    
    sounds(s,:) = stim.sound;
    events(s,:) = stim.sound2;
    stims{s,1} = stim;
    
    % disp(stim.params.nEvents)
end

save(fn, 'events', 'sounds', 'rates') 


%% Set 3
fn = 'stimData_s3_500x294.mat';
clear params

n = 500;
params.eventType = 'sine';
params.eventFreq = 22;
params.dispWarn = 0;
params.seedDebug = 0;
params.startBuff = 50;
params.endBuff = 400;
params.cutOff = 1150;
params.Fs = 256;
params.noiseMag = 0.0001;

sounds = NaN(n, round(params.Fs*params.cutOff/1000), 'double');
events = NaN(n, round(params.Fs*params.cutOff/1000), 'double');
stims = cell(n, 1);
rates = zeros(n, 1, 'int16');

for s = 1:n
    
    if ~mod(s, 100)
        disp([num2str(s), '/', num2str(n)])
    end
    
    rate = randi([6,16]);
    rates(s,1) = rate;
    
    params.startBuff = randi([0,50]);
    
    params.nEvents = rate;
    stim = TemporalStim(params);
    
    sounds(s,:) = stim.sound;
    events(s,:) = stim.sound2;
    stims{s,1} = stim;
    
    % disp(stim.params.nEvents)
end

save(fn, 'events', 'sounds', 'rates') 


%% Set 4
fn = 'stimData_s4_500x294.mat';
clear params

n = 500;
params.eventType = 'sine';
params.eventFreq = 220;
params.dispWarn = 0;
params.seedDebug = 0;
params.startBuff = 50;
params.endBuff = 400;
params.cutOff = 1150;
params.Fs = 1024;
params.noiseMag = 0.0001;

sounds = NaN(n, round(params.Fs*params.cutOff/1000), 'double');
events = NaN(n, round(params.Fs*params.cutOff/1000), 'double');
stims = cell(n, 1);
rates = zeros(n, 1, 'int16');

for s = 1:n
    
    if ~mod(s, 100)
        disp([num2str(s), '/', num2str(n)])
    end
    
    rate = randi([6,16]);
    rates(s,1) = rate;
    
    params.startBuff = randi([0,50]);
    
    params.nEvents = rate;
    stim = TemporalStim(params);
    
    sounds(s,:) = stim.sound;
    events(s,:) = stim.sound2;
    stims{s,1} = stim;
    
    % disp(stim.params.nEvents)
end

save(fn, 'events', 'sounds', 'rates') 


%% Set 5
fn = 'stimData_s5_500x400.mat';
clear params

n = 500;
params.eventType = 'sine';
params.eventFreq = 120;
params.dispWarn = 0;
params.seedDebug = 0;
params.startBuff = 50;
params.endBuff = 400;
params.cutOff = 1150;
params.Fs = 400;
params.noiseMag = 0.0001;

sounds = NaN(n, round(params.Fs*params.cutOff/1000), 'double');
events = NaN(n, round(params.Fs*params.cutOff/1000), 'double');
stims = cell(n, 1);
rates = zeros(n, 1, 'int16');

close all
example = TemporalStim(params);
example.plot()
drawnow

for s = 1:n
    
    if ~mod(s, 100)
        disp([num2str(s), '/', num2str(n)])
    end
    
    rate = randi([6,16]);
    rates(s,1) = rate;
    
    params.startBuff = randi([0,50]);
    
    params.nEvents = rate;
    stim = TemporalStim(params);
    
    sounds(s,:) = stim.sound;
    events(s,:) = stim.sound2;
    stims{s,1} = stim;
    
    % disp(stim.params.nEvents)
end

save(fn, 'events', 'sounds', 'rates') 


%% Set 6
fn = 'stimData_s6_500x400.mat';
clear params

n = 500;

params.type = 'Vis';
params.dispWarn = 0;
params.seedDebug = 0;
params.startBuff = 50;
params.endBuff = 400;
params.cutOff = 1150;
params.Fs = 400;
params.noiseMag = 0.0001;

close all
example = TemporalStim(params);
example.plot()
drawnow


sounds = NaN(n, round(params.Fs*params.cutOff/1000), 'double');
events = NaN(n, round(params.Fs*params.cutOff/1000), 'double');
stims = cell(n, 1);
rates = zeros(n, 1, 'int16');

for s = 1:n
    
    if ~mod(s, 100)
        disp([num2str(s), '/', num2str(n)])
    end
    
    rate = randi([6,16]);
    rates(s,1) = rate;
    
    params.startBuff = randi([0,50]);
    
    params.nEvents = rate;
    stim = TemporalStim(params);
    
    sounds(s,:) = stim.sound;
    events(s,:) = stim.sound2;
    stims{s,1} = stim;
    
    % disp(stim.params.nEvents)
end

save(fn, 'events', 'sounds', 'rates') 


%% Set 7
fn = 'stimData_s7_500x400.mat';
clear params

n = 500;
params.eventType = 'sine';
params.eventFreq = 120;
params.dispWarn = 0;
params.seedDebug = 0;
params.startBuff = 50;
params.endBuff = 400;
params.cutOff = 1150;
params.Fs = 400;
params.noiseMag = 0.0001;

sounds = NaN(n, round(params.Fs*params.cutOff/1000), 'double');
events = NaN(n, round(params.Fs*params.cutOff/1000), 'double');
stims = cell(n, 1);
rates = zeros(n, 1, 'int16');

close all
example = TemporalStim(params);
example.plot()
drawnow

for s = 1:n
    
    if ~mod(s, 100)
        disp([num2str(s), '/', num2str(n)])
    end
    
    rate = randi([3,8])*2;
    rates(s,1) = rate;
    
    params.startBuff = 60;
       
    params.nEvents = rate;
    stim = TemporalStim(params);
    
    sounds(s,:) = stim.sound;
    events(s,:) = stim.sound2;
    stims{s,1} = stim;
    
    % disp(stim.params.nEvents)
end

save(fn, 'events', 'sounds', 'rates') 


%% Set 8
fn = 'stimData_s8_500x400.mat';
clear params

n = 500;

params.type = 'Vis';
params.dispWarn = 0;
params.seedDebug = 0;
params.startBuff = 50;
params.endBuff = 400;
params.cutOff = 1150;
params.Fs = 400;
params.noiseMag = 0.0001;

close all
example = TemporalStim(params);
example.plot()
drawnow


sounds = NaN(n, round(params.Fs*params.cutOff/1000), 'double');
events = NaN(n, round(params.Fs*params.cutOff/1000), 'double');
stims = cell(n, 1);
rates = zeros(n, 1, 'int16');

for s = 1:n
    
    if ~mod(s, 100)
        disp([num2str(s), '/', num2str(n)])
    end
    
    rate = randi([3,8])*2;
    rates(s,1) = rate;
    
    params.startBuff = 60;
    
    params.nEvents = rate;
    stim = TemporalStim(params);
    
    sounds(s,:) = stim.sound;
    events(s,:) = stim.sound2;
    stims{s,1} = stim;
    
    % disp(stim.params.nEvents)
end

save(fn, 'events', 'sounds', 'rates') 


%% Set 9 AV
% Matched rates
% No start buff variation

fn = 'stimData_AV_s9_500x400.mat';
clear params

n = 500;

params.dispWarn = 0;
params.seedDebug = 0;
params.startBuff = 50;
params.endBuff = 400;
params.cutOff = 1300;
params.Fs = 400;
params.noiseMag = 0.0001;
params.startBuff = 60;

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


for s = 1:500
    
    if ~mod(s, 100)
        disp([num2str(s), '/', num2str(n)])
    end
    
    rate = randi([3,8])*2;
    
    params.type = 'Aud';
    params.eventType = 'sine';
    ratesA(s,1) = rate;

    params.nEvents = rate;
    stimA = TemporalStim(params);
    
    soundsA(s,:) = stimA.sound;
    eventsA(s,:) = stimA.sound2;
    stimsA{s,1} = stimA;
    
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


%% Set 10 AV
% Same as above, more
% Matched rates
% Non-matched distributions
% No start buff variation

n = 10000;
fn = ['stimData_AV_s10_', num2str(n), 'x400.mat'];
clear params

params.dispWarn = 0;
params.seedDebug = 0;
params.startBuff = 50;
params.endBuff = 400;
params.cutOff = 1300;
params.Fs = 400;
params.noiseMag = 0.0001;
params.startBuff = 60;

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
    
    rate = randi([3,8])*2;
    
    params.type = 'Aud';
    params.eventType = 'sine';
    ratesA(s,1) = rate;

    params.nEvents = rate;
    stimA = TemporalStim(params);
    
    soundsA(s,:) = stimA.sound;
    eventsA(s,:) = stimA.sound2;
    stimsA{s,1} = stimA;
    
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


%% Set 11 AV
% Same as above, more
% Matched rates
% Non-matched distributions
% Variable start buff, unmatched

n = 10000;
fn = ['stimData_AV_s11_', num2str(n), 'x400.mat'];
clear params

params.dispWarn = 0;
params.seedDebug = 0;
params.startBuff = 50;
params.endBuff = 500;
params.cutOff = 1300;
params.Fs = 400;
params.noiseMag = 0.0001;
params.startBuff = 60;

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

