function main_O2_ode15s_simple_model(analysis_name,randomseed)

result_name = 'Optimal_parameters_20201209T195254';

if nargin == 1
    randomseed = 'shuffle';
    
else
    rng(randomseed);
end

rng(randomseed);
file1 = [analysis_name,'/O2_exp.csv'];
C = readtable(file1);
raw_data = C.O2;
%average_data = sum(raw_data)/length(raw_data); 
%O2 = raw_data/average_data; 
O2_exp = raw_data;

file1 = [analysis_name,'/FvFm_exp.csv'];
C = readtable(file1);
FvFm_exp = C.FqFm;


file1 = [analysis_name,'/experimental_parameters.xls'];
tbl1 = readtable(file1);
n_flashes = tbl1.n_flashes;
flash_duration = tbl1.flash_duration;
flash_interval = tbl1.flash_interval;
train_interval = tbl1.train_interval;
n_trains = tbl1.n_trains;
% 
% n_flashes = 50;
% flash_duration = 5e-5;
% % intervals = [1];
% intervals = 1;
% train_interval = .0001;

file1 = [analysis_name,'/LaiskConstants.xls'];
tablek = readtable(file1);
indepk = find(tablek.independent);
lbk = tablek.lb(indepk);
ubk = tablek.ub(indepk);
knames = tablek.name;
k_std = tablek.base_val;

file2 = [analysis_name,'/LaiskY.xls'];
tabley = readtable(file2);
indepy = find(tabley.independent);
y_std = tabley.base_val;
lby = tabley.lb(indepy);
uby = tabley.ub(indepy);
yr = lby + (uby-lby).*rand(length(indepy),1);
kr = lbk + (ubk-lbk).*rand(length(indepk),1);

%yr = yr+ yr.*rand(length(yr),1)*.5 - yr.*rand(length(yr),1)*.5;
% kr = k_std(indepk);
% yr = y_std(indepy);
%kr = kr+ kr.*rand(length(kr),1)*.5 - kr.*rand(length(kr),1)*.5;

indep_ps2 = find(contains(tabley.name(indepy), 'Y'));
Ay = zeros(1,length(yr));
Ay(indep_ps2) = 1;
by = 1;

Ak = zeros(1,length(indepk));
Aeq = [Ay,Ak];
beq = 1;

x0 = [yr;kr];

lb = [reshape(lby,[],1); reshape(lbk,[],1)];
ub = [reshape(uby,[],1); reshape(ubk,[],1)];

Ynames = tabley.name;
file3 = [analysis_name,'/LaiskReactions.xlsx'];
[~,Rknames] = xlsread(file3);

PFD = find(strcmp(knames, 'PFD')); 
a2 = find(strcmp(knames, 'a2'));
b1d = find(strcmp(knames, 'b1d'));
b2d = find(strcmp(knames, 'b2d'));
Chl = find(strcmp(knames, 'Chl')); 
CytfT = find(strcmp(knames, 'CytfT')); 
FDT = find(strcmp(knames, 'FDT')); 
jd = find(strcmp(knames, 'jd')); 
kb6f = find(strcmp(knames, 'kb6f')); 
kcytf = find(strcmp(knames, 'kcytf')); 
kf = find(strcmp(knames, 'kf')); 
kfd = find(strcmp(knames, 'kfd')); 
kfx = find(strcmp(knames, 'kfx')); 
kn = find(strcmp(knames, 'kn')); 
kp = find(strcmp(knames, 'kp'));
kpc = find(strcmp(knames, 'kpc'));
kr = find(strcmp(knames, 'kr')); 
kE1 = find(strcmp(knames, 'kE1'));
kEb6f = find(strcmp(knames, 'kEb6f'));
kEcytf = find(strcmp(knames,'kEcytf'));
kEfx = find(strcmp(knames,'kEfx'));
kEpc = find(strcmp(knames,'kEpc'));
Labs = find(strcmp(knames,'Labs'));
oqd = find(strcmp(knames,'oqd'));
oqr = find(strcmp(knames,'oqr'));
PCT = find(strcmp(knames,'PCT'));
PQT = find(strcmp(knames,'PQT'));
PSU1 = find(strcmp(knames,'PSU1'));
PSU2 = find(strcmp(knames,'PSU2'));
rqd = find(strcmp(knames,'rqd'));
rqr = find(strcmp(knames,'rqr'));
b1r = find(strcmp(knames,'b1r'));
kq = find(strcmp(knames,'kq')); 
P700T = find(strcmp(knames, 'P700T'));
FXT = find(strcmp(knames, 'FXT')); 
 
YoPoAo = find(strcmp(Ynames,'YoPoAo')); 
YoPoAoBoo = find(strcmp(Ynames,'YoPoAoBoo')); 
YoPrAo = find(strcmp(Ynames,'YoPrAo')); 
YoPoAr = find(strcmp(Ynames,'YoPoAr')); 
YoPrAoBoo = find(strcmp(Ynames,'YoPrAoBoo')); 
YoPoArBoo = find(strcmp(Ynames,'YoPoArBoo')); 
YoPoAoBro = find(strcmp(Ynames,'YoPoAoBro')); 
YrPrAo = find(strcmp(Ynames,'YrPrAo')); 
YoPrAr = find(strcmp(Ynames,'YoPrAr')); 
YrPrAoBoo = find(strcmp(Ynames,'YrPrAoBoo')); 
YoPrArBoo = find(strcmp(Ynames,'YoPrArBoo')); 
YoPrAoBro = find(strcmp(Ynames,'YoPrAoBro')); 
YoPoArBro = find(strcmp(Ynames,'YoPoArBro')); 
YoPoAoBrr = find(strcmp(Ynames,'YoPoAoBrr')); 
YrPrAr = find(strcmp(Ynames,'YrPrAr')); 
YrPrArBoo = find(strcmp(Ynames,'YrPrArBoo')); 
YoPrArBro = find(strcmp(Ynames,'YoPrArBro')); 
YoPrAoBrr = find(strcmp(Ynames,'YoPrAoBrr')); 
YrPrAoBro = find(strcmp(Ynames,'YrPrAoBro')); 
YoPoArBrr = find(strcmp(Ynames,'YoPoArBrr')); 
YrPrArBro = find(strcmp(Ynames,'YrPrArBro')); 
YrPrAoBrr = find(strcmp(Ynames,'YrPrAoBrr')); 
YoPrArBrr = find(strcmp(Ynames,'YoPrArBrr')); 
YrPrArBrr = find(strcmp(Ynames,'YrPrArBrr')); 
PQH2 = find(strcmp(Ynames,'PQH2')); 
PQ = find(strcmp(Ynames,'PQ')); 
Cytfr = find(strcmp(Ynames,'Cytfr')); 
PCr = find(strcmp(Ynames,'PCr')); 
P700r = find(strcmp(Ynames,'P700r')); 
FXr = find(strcmp(Ynames,'FXr')); 
FDr = find(strcmp(Ynames,'FDr')); 
PCo = find(strcmp(Ynames, 'PCo'));
P700o = find(strcmp(Ynames, 'P700o'));
FXo = find(strcmp(Ynames, 'FXo'));
FDo = find(strcmp(Ynames, 'FDo'));
Cytfo = find(strcmp(Ynames, 'Cytfo')); 
O2 = find(strcmp(Ynames, 'O2')); 
Fl = find(strcmp(Ynames, 'Fl')); 
 
% PS1T = k(a2)*k(Chl)/k(PSU2); 
% PS2T = (1-k(a2))*(k(Chl)/k(PSU1));
% PS2TR = (1-k(a2))*(k(Chl)/k(PSU1));

% PS1T = 1; 
% PS2T = 1;
% n1 = k(PFD)*k(Labs)*(1-k(a2))/PS1T;
% n2 = k(PFD)*k(Labs)*k(a2)/PS2T; 
% y0(P700r) = PS1T/PS2T;
% y0(FXo) = PS1T/PS2T;



% 
% k(oqr) = k(oqr);
% k(rqr) = k(rqr);
% k(kb6f) = k(kb6f);
% k(kcytf) = k(kcytf);
% k(kpc) = k(kpc);
% k(kfx) = k(kfx);
% k(P700T) = PS1T/PS2T; 
% k(FXT) = PS1T/PS2T; 
% k(kfd) = k(kfd);
mult1 = find(strcmp(knames,'n2*kp/(1+kp+kn+kr)'));
mult2 = find(strcmp(knames,'n2*kp/(1+kp+kn)')); 
Div1 = find(strcmp(knames,'kpc/kEpc'));
Div2 = find(strcmp(knames,'kcytf/kEcytf'));
Div3 = find(strcmp(knames,'kfx/kEfx'));
Div4 = find(strcmp(knames,'kb6f/kEb6f'));
n1idx = find(strcmp(knames,'n1'));

% mult1Val = n2*k(kp)/(1+k(kp)+k(kn)+k(kr));
% mult2Val = n2*k(kp)/(1+k(kp)+k(kn));
% Div1Val = k(kpc)/k(kEpc);
% Div2Val = k(kcytf)/k(kEcytf);
% Div3Val = k(kfx)/k(kEfx);
% Div4Val = k(kb6f)/k(kEb6f);
% 
% k(mult1) = mult1Val;
% k(mult2) = mult2Val;
% k(Div1) = Div1Val;
% k(Div2) = Div2Val;
% k(Div3) = Div3Val; 
% k(Div4) = Div4Val;
% k(n1idx) = n1;

% idcs = struct;
kidcs.PFD = PFD;
kidcs.Labs = Labs;
kidcs.a2 = a2;
kidcs.PSU2 = PSU2;
kidcs.Chl = Chl;
kidcs.PSU1 = PSU1;
kidcs.kp = kp;
kidcs.kn = kn;
kidcs.kr = kr;
kidcs.Div1 = Div1;
kidcs.Div2 = Div2;
kidcs.Div3 = Div3;
kidcs.Div4 = Div4;
kidcs.mult1 = mult1;
kidcs.mult2 = mult2;
kidcs.n1idx = n1idx;
kidcs.kpc = kpc;
kidcs.kEpc = kEpc;
kidcs.kcytf = kcytf;
kidcs.kEcytf = kEcytf;
kidcs.kfx = kfx;
kidcs.kEfx = kEfx;
kidcs.kb6f = kb6f;
kidcs.kEb6f = kEb6f;
kidcs.kf = kf;


[species,S,rate_inds] = Laisk_read_excel_model(analysis_name);
species_idcs = zeros(length(species),1);
% yinitial = zeros(length(y0),1);
for i = 1:length(Ynames)
    index = find(strcmp(species,Ynames(i)));
%     yinitial(index) = y0(i);
    species_idcs(index) = i;
end
PSIidcs = zeros(2,1);
% PSIidcs(1) = find(strcmp(species,'P700r'));%- Commented out for the simpler model
% up to PQ
% PSIidcs(2) = find(strcmp(species,'FXr')); %- Commented out for the simpler model
% up to PQ

[kconst] = LaiskKconstantsReadTable(analysis_name);


kn = find(strcmp(knames,'kn'));
kp = find(strcmp(knames,'kp'));
kr = find(strcmp(knames,'kr')); 
kq = find(strcmp(knames,'kq')); 
Fluorescence_k_idcs = [kn;kp;kr;kq];

yopoax = find(contains(species,'YoPoA'));
yoprao = find(contains(species,'YoPrAo'));
yoprar = find(contains(species,'YoPrAr'));
yrprao = find(contains(species,'YrPrAo'));
yrprar = find(contains(species,'YrPrAr'));
Fluorescence_y_inds = {yopoax;yoprao;yoprar;yrprao;yrprar};


O2_exp = [156.18408
563.45215
918.83545
729.83887
426.49414
504.04785
713.70605
704.78516
539.23584
532.05811
643.85986
671.5625
588.23242
560.5127
617.14844
648.76465
607.76611
582.98584
609.74854
633.93066
615.84961
596.29883
607.32178
625.36865
618.12256
604.72412
608.05664
619.54102
617.2168
608.74023
609.7998
617.81494
618.00293
611.93604
611.95312
616.0376
616.85791
614.58496
613.25195
615.71289
616.25977
614.82422
612.82471
615.72998
617.42187
616.37939
615.49072
617.33643
617.83203
617.60986]; %nanoDBMIB100

O2_exp = [42.7417
464.16016
1026.51855
741.45996
429.68994
446.95068
669.35791
690.7373
570.4248
526.33301
596.94824
635.81055
596.93115
567.62207
588.69385
609.49219
597.29004
581.55029
588.54004
597.81982
594.31641
585.54932
589.22363
596.50391
593.44482
588.83057
589.97559
593.05176
592.28271
589.90723
590.77881
593.61572
593.13721
592.23145
591.85547
594.26514
593.71826
592.04346
592.06055
592.33398
593.35937
592.50488
593.20557
594.02588
594.04297
593.76953
594.81201
594.74365
594.96582
593.66699]; %nanoDBMIB50

O2_exp = [23.75488
53.42285
1096.39893
1024.16016
655.01953
422.64893
665.35889
804.55566
727.10449
588.96729
601.08398
667.10205
684.39697
644.95361
627.21436
645.96191
670.48584
663.85498
658.98437
661.34277
671.51123
672.82715
670.22949
667.0166
670.2124
671.2207
672.04102
668.14453
669.64844
670.53711
672.36572
671.88721
672.92969
673.45947
675.21973
674.91211
677.21924
676.604
677.42432
676.09131
677.16797
676.31348
677.4585
677.76611
679.44092
678.85986
679.71436
679.69727
680.67139
680.3125]; %nanocontrol

O2_exp = [26.98486
44.7583
1016.53809
964.03809
619.48975
394.62158
594.19678
718.25195
659.89014
545.16602
549.13086
607.97119
632.99072
606.36475
589.83887
605.86914
626.54785
625.7959
617.31934
617.90039
624.51416
624.01855
620.15625
614.87549
616.39648
617.98584
618.95996
616.96045
619.01123
620.39551
621.50635
621.01074
623.07861
621.98486
623.33496
622.49756
624.12109
622.71973
622.83936
622.17285
623.35205
623.40332
622.58301
623.50586
625.21484
624.46289
625.52246
624.71924
626.00098
624.08691]; %nanoDBMIB2.5

O2_exp = [27.33541
128.4406
967.29349
831.64182
511.50319
369.96594
581.72161
678.17502
602.84632
510.3861
543.14024
598.36128
595.65191
557.07055
555.15315
574.91902
581.06302
568.81669
564.92354
570.44231
574.11038
570.51734
569.2752
570.21723
571.83451
570.43397
571.08422
570.45065
570.90082
570.40897
570.30893
571.00086
571.97623
570.21723
570.10051
569.65034
570.3256
569.78373
570.60904
570.04216
571.82617
570.80078
570.45898
570.25891
570.62571
569.80874
569.99214
570.34227
571.1926
570.10885]; %nanoDMBIB25


O2_exp = [0.0050
    0.1431
    1.7243
    1.5172
    0.9829
    0.6837
    1.0088
    1.1670
    1.1111
    0.9692
    0.9831
    1.0476
    1.0697
    1.0300
    1.0115
    1.0170
    1.0339
    1.0293
    1.0261
    1.0203
    1.0256
    1.0233
    1.0203
    1.0201];

O2_exp = O2_exp/mean(O2_exp);

n_flashes = length(O2_exp);

fun = @(x) calc_sqerror_simple_model_O2(x,...Set of parameters including k and yinitial
                    n_trains, n_flashes, flash_duration, flash_interval, train_interval, ... Experimental parameters
                    Fluorescence_k_idcs, Fluorescence_y_inds,... Indeces to calculate Fluorescence
                    kidcs, PSIidcs, ... all indices needed to calculate FvFm and prepare the variables
                    tablek, tabley,... information on the k and y variables
                    kconst, rate_inds, S, species, knames, Rknames, species_idcs,...
                    FvFm_exp,O2_exp);
              
opts = optimoptions(@fmincon,'Algorithm','interior-point','MaxFunEvals',10000, 'Display', 'iter');


[xopt, fval, exitflag, output,lambda, grad,hessian] = fmincon(fun, ...
                      x0, [], [], [], [], lb, ub, [], opts);                
                
%problem = createOptimProblem('fmincon',...
%                            'objective', fun, ...
%                            'x0', x0, 'lb', lb, 'ub', ub, 'options', opts);                     
%ms = MultiStart('UseParallel', true);
% To change number of runs change the number "n" after problem below. 
%n = 1;
%[xopt,fval,exitflag,output] = run(ms,problem,n);

O2_sim_opt = calc_O2_simple_model_ode15s(xopt, n_trains, n_flashes, ...
                    flash_duration, flash_interval, train_interval, ... 
                    Fluorescence_k_idcs, Fluorescence_y_inds, kidcs, ... 
                    tablek, tabley, kconst, rate_inds, ... 
                    S, species, species_idcs, Rknames);


if ~exist('results', 'dir')
    mkdir('results')
end
if ~exist(['results/' analysis_name], 'dir')
    mkdir('results', analysis_name);                 
end
save(['results/', analysis_name '/Optimal_parameters_' datestr(now, 30)],...
    'xopt', 'fval', 'exitflag', 'output',...optimization parameters
    'n_trains', 'n_flashes', 'flash_duration', 'flash_interval', 'train_interval', ... Experimental parameters
                    'kidcs', 'PSIidcs',... all indices needed in to calculate FvFm and prepare the variables
                    'tablek', 'tabley', 'Rknames',... information on the k and y variables
                    'kconst', 'rate_inds', 'S', 'species', 'knames', 'species_idcs',...
                    'FvFm_exp', 'O2_sim_opt','O2_exp')


end





