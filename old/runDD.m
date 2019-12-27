%% Load dataset

a = load('stimData_AV_s14_20000x400.mat');
n = size(a.soundsA,1);
n = 2000;


%% Predict on each

decs = NaN(n, 3);

% General prarms for DD
params.aLam = 0.99; 
params.contPlot = 0;

for i = 1:n
   
    if ~mod(i, 100)
        disp(['Iteration: ', num2str(i), '/', num2str(n)])
    end
    
    % Run A decision
    stim1.delta1 = abs(a.soundsA(i,:));
    mods.DD1 = DD(params, stim1);
    mods.DD1 = mods.DD1.run;
    decs(i,1) = mods.DD1.dec;
    
    % Run V decision
    stim1.delta1 = a.soundsV(i,:);
    mods.DD2 = DD(params, stim1);
    mods.DD2 = mods.DD2.run;
    decs(i,2) = mods.DD2.dec;
    
    % Run AV decision
    params.treshold = 10; % Not used
    mulMod = MultiDD(mods, params);
    decs(i,3) = mulMod.finalDecMag;
end


%% Tabulate results

results = table(a.ratesA(1:n), decs(:,1), decs(:,2), decs(:,3));
results.Properties.VariableNames = {'Rate', 'decVarA', 'decVarV', 'decVarAV'};

% Normalise to approximate a threshold
results.decA = results.decVarA>mean(results.decVarA);
results.decV = results.decVarV>mean(results.decVarV);
results.decAV = results.decVarAV>mean(results.decVarAV);

fastProp = grpstats(results, 'Rate');


%% Plot fast prop

ffitA = fitPsyche(single(fastProp.Rate), fastProp.mean_decA, 'GLM');
ffitV = fitPsyche(single(fastProp.Rate), fastProp.mean_decV, 'GLM');
ffitAV = fitPsyche(single(fastProp.Rate), fastProp.mean_decAV, 'GLM');

figure
plotPsyche(ffitA)
hold on
plotPsyche(ffitV)
plotPsyche(ffitAV)

ylabel('Fast prop.')
xlabel('Rate')
legend({'A', 'A fit', 'V', 'V fit', 'AVs', 'AVs fit'})
