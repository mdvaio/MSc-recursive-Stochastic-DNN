clc 
clear all
close all
load ./../Dados/Xt_Pocos.mat
%                    1           2       3       4      5        6           
% DATA_Label = ["ElapsedTime", "RPM", "SWOB", "STOR", "ROP", "SS_STOR",
%                    8     9      10         11        12        13        14
%                 "SPPA", "SS", "score1", "score2", "score3", "score4", "score5"]

% PARA = Xt_PocoA_06_12_50(1:23466, :);
% PARB = Xt_PocoA_06_12_50(23500:end, :);
% PBR1 = Xt_PocoBr1_06_12_50;
% PBR2 = Xt_PocoBr2_06_12_50;

% PARA = Xt_PocoA_20_40_50(1:22449, :);
% PARB = Xt_PocoA_20_40_50(20500:end, :);
% PBR1 = Xt_PocoBr1_20_40_50;
% PBR2 = Xt_PocoBr2_20_40_50;

PARA = Xt_PocoA_50_100_50(1:20408, :);
PARB = Xt_PocoA_50_100_50(20458:end, :);
PBR1 = Xt_PocoBr1_50_100_50;
PBR2 = Xt_PocoBr2_50_100_50;

% PARA = Xt_PocoA_50_100_50mm(1:20408, :);
% PARB = Xt_PocoA_50_100_50mm(20458:end, :);
% PBR1 = Xt_PocoBr1_50_100_50mm;
% PBR2 = Xt_PocoBr2_50_100_50mm;

% PARA = Xt_PocoA_50_30_50(1:20243, :);
% PARB = Xt_PocoA_50_30_50(20244:end, :);
% PBR1 = Xt_PocoBr1_50_30_50;
% PBR2 = Xt_PocoBr2_50_30_50;
%%


X1 = PARA(:, 1:end-1);
X2 = PARB(:, 1:end-1);
X3 = PBR1(:, 1:end-1);
X4 = PBR2(:, 1:end-1);

SS1 = PARA(:, end);
SS2 = PARB(:, end);
SS3 = PBR1(:, end);
SS4 = PBR2(:, end);


XX = [X1; X2; X3; X4]; % TIME RPM SWOB STOR ROP SS_STOR SPPA
SS = [SS1; SS2; SS3; SS4]; % SS
XX = [XX SS]; % TIME RPM SWOB STOR SS_STOR ROP SPPA SS

      
X = XX(:,[2 3 4 5 6]); % RPM SWOB STOR SS_STOR ROP 



dimensoes = [length(SS1) length(SS2) length(SS3) length(SS4)];
% dimensoes = [length(SS1)-seq_len-70 length(SS2) length(SS3) length(SS4)];
size_pocos = cumsum(dimensoes);
% dimensoes = [20127 length(SS1)-20127];

norm_log = 1;
teste = 1;

% seq_len = 6;
% for i=seq_len+1:length(X)
%     XX2(i-seq_len,:) = [X(i,:) X(i-1,:) X(i-2,:) X(i-3,:) X(i-4,:) X(i-5,:)];
% end

% X_norm_controlar = 1;
% RPM_Norm = 215;
% SWOB_Norm = 85;
% STOR_Norm = 55;
% ROP_Norm = 65;
% SS_STOR_Norm = 4.3;
% 
% clearvars X_new1 X_new2 X_new3 X_new4 X_new5

% for i=seq_len+1:length(X)
%     X_new1(i-seq_len,:) = [X(i,1) X(i-1,1) X(i-2,1) X(i-3,1) X(i-4,1) X(i-5,1)];
%     X_new2(i-seq_len,:) = [X(i,2) X(i-1,2) X(i-2,2) X(i-3,2) X(i-4,2) X(i-5,2)]; 
%     X_new3(i-seq_len,:) = [X(i,3) X(i-1,3) X(i-2,3) X(i-3,3) X(i-4,3) X(i-5,3)]; 
%     X_new4(i-seq_len,:) = [X(i,4) X(i-1,4) X(i-2,4) X(i-3,4) X(i-4,4) X(i-5,4)]; 
%     X_new5(i-seq_len,:) = [X(i,5) X(i-1,5) X(i-2,5) X(i-3,5) X(i-4,5) X(i-5,5)];
%     
%     if X_norm_controlar == 1        
%    
% %         X_new1(i-seq_len,:) = (X_new1(i-seq_len,:) - mean(X_new1(i-seq_len,:))) / std(X_new1(i-seq_len,:));
% %         X_new2(i-seq_len,:) = (X_new2(i-seq_len,:) - mean(X_new2(i-seq_len,:))) / std(X_new2(i-seq_len,:));
% %         X_new3(i-seq_len,:) = (X_new3(i-seq_len,:) - mean(X_new3(i-seq_len,:))) / std(X_new3(i-seq_len,:));
% %         X_new4(i-seq_len,:) = (X_new4(i-seq_len,:) - mean(X_new4(i-seq_len,:))) / std(X_new4(i-seq_len,:));
% %         X_new5(i-seq_len,:) = (X_new5(i-seq_len,:) - mean(X_new5(i-seq_len,:))) / std(X_new5(i-seq_len,:));  
% 
% %         X_new1(i-seq_len,:) = (X_new1(i-seq_len,:)) / RPM_Norm;
% %         X_new2(i-seq_len,:) = (X_new2(i-seq_len,:)) / SWOB_Norm;
% %         X_new3(i-seq_len,:) = (X_new3(i-seq_len,:)) / STOR_Norm;
% %         X_new4(i-seq_len,:) = (X_new4(i-seq_len,:)) / ROP_Norm;
% %         X_new5(i-seq_len,:) = (X_new5(i-seq_len,:)) / SS_STOR_Norm; 
%     
%     end
% 
% end
% XX2 = [X_new1 X_new2 X_new3 X_new4 X_new5];
% X_norm_controlado = XX2;
% X = XX2;
% XX = XX(seq_len+1:end,:);


fig1 = figure('units','normalized','outerposition',[0 0.035 1/3 0.92]);
fig2 = figure('units','normalized','outerposition',[1/3 0.035 1/3 0.92]);
fig3 = figure('units','normalized','outerposition',[2/3 0.035 1/3 0.92]);
%%   -------  

count = 0; count_ant = 1;

% figure
figure(fig1)
% subplot(2,1,teste)

for i=1:length(size_pocos)
    
    count = size_pocos(i);
   
    Proj1 = XX(count_ant:count,2);
    Proj2 = XX(count_ant:count,3);
    Proj3 = XX(count_ant:count,4);
    
    scatter3(Proj1,Proj2, Proj3, [20], XX(count_ant:count,8), 'filled')
    hold on
    count_ant = count;
end
hold off
colorbar('Ticks', [0.5, 0.8 1.2 2.25 3.5]); caxis([0.5 2.7])
xlabel('RPM'); ylabel('SWOB'); zlabel('STOR')

nstd_min = 4;
nstd_max = 3;
xmin = mean(XX(:,2)) - nstd_min*std(XX(:,2)); xmax = mean(XX(:,2)) + nstd_max*std(XX(:,2));
ymin = mean(XX(:,3)) - nstd_min*std(XX(:,3)); ymax = mean(XX(:,3)) + nstd_max*std(XX(:,3));
zmin = mean(XX(:,4)) - nstd_min*std(XX(:,4)); zmax = mean(XX(:,4)) + nstd_max*std(XX(:,4));
set(gca,'XLim',[xmin xmax],'YLim',[0 ymax],'ZLim',[zmin zmax])

colormap(jet);
grid on
title('Mapa de SS GERAL')

%%   -------   Calcula PCA - Geral
clearvars X_norm coeff score latent tsquared explained mu
clc
count = 0;
count_ant = 1;

% X=XX2;

if norm_log == 1
    X_norm = log(X) - log(std(X));
elseif norm_controlada == 1
    X_norm = X_norm_controlado;
else
    X_norm = (X - mean(X))./std(X);
end

[coeff,score,latent,tsquared,explained,mu_] = pca(X_norm);

% [coeff,score,latent,tsquared,explained,mu_] = pca(X);
XX = [XX score];

% figure
figure(fig2)
% subplot(2,1,teste)
for i=1:length(size_pocos)
    
    count = size_pocos(i);
   
    Score1 = score(count_ant:count,1);
    Score2 = score(count_ant:count,2);
    Score3 = score(count_ant:count,3);
    
    scatter3(Score1,Score2, Score3, [20], SS(count_ant:count), 'filled')
    hold on
    count_ant = count;
end
hold off
colorbar('Ticks', [0.5, 0.8 1.2 2.25 3.5]); caxis([0.5 2.7])
xlabel(string(explained(1))); ylabel(string(explained(2))); zlabel(string(explained(3)))
colormap(jet);
grid on

if norm_log == 0
    title('Calcula PCA Geral')
else
    title('Calcula PCA Geral (log)')
end



nstd_min = 3;
nstd_max = 3;
xmin = mean(score(:,1)) - nstd_min*std(score(:,1)); xmax = mean(score(:,1)) + nstd_max*std(score(:,1));
ymin = mean(score(:,2)) - nstd_min*std(score(:,2)); ymax = mean(score(:,2)) + nstd_max*std(score(:,2));
zmin = mean(score(:,3)) - nstd_min*std(score(:,3)); zmax = mean(score(:,3)) + nstd_max*std(score(:,3));
set(gca,'XLim',[xmin xmax],'YLim',[1*ymin 1.2*ymax],'ZLim',[zmin zmax])

xlabel('PC1'); ylabel('PC2'); zlabel('PC3')



%%   -------   Calcula PCA - Geral por partes
% clearvars X_norm coeff score latent tsquared explained mu
% clc
% count = 0;
% count_ant = 1;
% 
% if norm_log == 1
%     X_norm = log(X) - log(std(X));
% else
%     X_norm = (X - mean(X))./std(X);
% end
% 
% 
% [coeff,score,latent,tsquared,explained,mu_] = pca(X_norm);
% 
% figure
% figure(fig2)
% subplot(2,1,teste)
% 
% clearvars Coeff Explained
% 
% for i=1:length(size_pocos)
%     
%     count = size_pocos(i);
%     [coeff,score,latent,tsquared,explained,mu_] = pca(X_norm(count_ant:count,:));
%     
%     Score1 = score(:,1);
%     Score2 = score(:,2);
%     Score3 = score(:,3);
%     
%     Explained(:,:,i) = explained;
%     Coeff(:,:,i) = coeff;
%     
%     scatter3(Score1,Score2, Score3, [20], SS(count_ant:count), 'filled')
%     hold on
%     count_ant = count;
% end
% hold off
% colorbar('Ticks', [0.5, 0.8 1.2 2.25 3.5]); caxis([0.5 2.7])
% xlabel(string(explained(1))); ylabel(string(explained(2))); zlabel(string(explained(3)))
% colormap(jet);
% grid on
% 
% if norm_log == 0
%     title('Calcula PCA Geral')
% else
%     title('Calcula PCA Geral (log)')
% end
% 
% 
% 
% nstd_min = 3;
% nstd_max = 3;
% xmin = mean(score(:,1)) - nstd_min*std(score(:,1)); xmax = mean(score(:,1)) + nstd_max*std(score(:,1));
% ymin = mean(score(:,2)) - nstd_min*std(score(:,2)); ymax = mean(score(:,2)) + nstd_max*std(score(:,2));
% zmin = mean(score(:,3)) - nstd_min*std(score(:,3)); zmax = mean(score(:,3)) + nstd_max*std(score(:,3));
% set(gca,'XLim',[xmin xmax],'YLim',[1*ymin 1.2*ymax],'ZLim',[zmin zmax])
% 
% xlabel('PC1'); ylabel('PC2'); zlabel('PC3')
% 



%%    ------   Calcula PCA com SS < que padrao
clearvars SS_m1 X_m1 coeff_m1 score_m1 latent_m1 tsquared_m1 explained_m1 mu_m1 X_m1_norm
clc
count = 1;
%SS = SS(seq_len+1:end,:);

% for i=1:size_pocos(1)

for i=1:size_pocos(1)
    if (SS(i) <= 1) && (min(X(i, :)) > 0) && (SS(i) > 0)
        X_m1(count, :) = X(i, :);
        SS_m1(count, :) = SS(i, :);
        count = count + 1;
    end
end

figure
scatter(X_m1(:,1), X_m1(:,2), [20], SS_m1, 'filled') 

if norm_log == 1
    X_m1_norm = log(X_m1)-log(std(X_m1));
elseif norm_controlada == 1
    X_m1_norm = X_norm_controlado;
else
    X_m1_norm = (X_m1-mean(X_m1))./std(X_m1);
end



X_m1_norm = X_m1;
[coeff_m1,score_m1,latent_m1,tsquared_m1,explained_m1,mu_m1] = pca(X_m1_norm);

% figure
% scatter3(score_m1(:, 1), score_m1(:, 2), score_m1(:, 3), [20], SS_m1, 'filled')
% colorbar('Ticks', [0.2, 0.4 0.6 0.8 1]); caxis([0.1 1]);
% colormap(jet);
% grid on

%   -------   Projeta dados em determinado PC

if norm_log == 1
    X_proj_norm = log(X) - log(std(X_m1));
    
else
    X_proj_norm = (X-mean(X_m1))./std(X_m1);
end

Lamb = coeff_m1;

% Lamb = [0.0486    0.1674   -0.0750    0.7590    0.6228;
%        -0.2564    0.2205    0.9392    0.0586    0.0024;
%        -0.2610    0.3600   -0.1196   -0.6064    0.6483;
%        -0.3160    0.7733   -0.2780    0.1816   -0.4380;
%         0.8740    0.4425    0.1435   -0.1405    0.0013];
    
% Lamb = [0.006    0.2674   -0.0750    0.7590    0.6228;
%        -0.4564    0.00    0.9392    0.0586    0.0024;
%        -0.00    0.4600   -0.1196   -0.6064    0.6483;
%        -0.000    0.6   -0.2780    0.1816   -0.4380;
%         0.80    0.0    0.1435   -0.1405    0.0013];
% 
% for i=1:size(Lamb, 2)
%     Lamb(:,i) = Lamb(:,i)/norm(Lamb(:,i));
% end
% Lamb
Proj = X_proj_norm*Lamb;
XX = [XX Proj];

count = 0;
count_ant = 1;

% figure
figure(fig3)
% subplot(2,1,teste)


for i=1:length(size_pocos)
    count = size_pocos(i);
   
    Proj1 = Proj(count_ant:count,1);
    Proj2 = Proj(count_ant:count,2);
    Proj3 = Proj(count_ant:count,3);
    hold on
    scatter3(Proj1,Proj2, Proj3, [20], SS(count_ant:count), 'filled')
    count_ant = count;
end
hold off

colorbar('Ticks', [0.5, 0.8 1.2 2.25 3.5]); caxis([0.5 2.7])
colormap(jet);
grid on
title('Calcula Projeção PCA SS<1 (PoçoA RochaA)')

if norm_log == 0
    title('Calcula Projeção PCA SS<1 (PoçoA RochaA)')
else
    title('Calcula Projeção PCA SS<1 (PoçoA RochaA) (log)')
end

nstd_min = 4;
nstd_max = 4;
xmin = mean(Proj(:,1)) - nstd_min*std(Proj(:,1)); xmax = mean(Proj(:,1)) + nstd_max*std(Proj(:,1));
ymin = mean(Proj(:,2)) - nstd_min*std(Proj(:,2)); ymax = mean(Proj(:,2)) + nstd_max*std(Proj(:,2));
zmin = mean(Proj(:,3)) - nstd_min*std(Proj(:,3)); zmax = mean(Proj(:,3)) + nstd_max*std(Proj(:,3));
set(gca,'XLim',[xmin xmax],'YLim',[1*ymin 0.6*ymax],'ZLim',[zmin zmax])
xlabel('PPC1'); ylabel('PPC2'); zlabel('PPC3')


%%

% ss = XX(:, 8);
% SSmm_l = movavg(ss, 'linear', 20);
% SSmm_s = movavg(ss, 'square', 20);
% SSmm_sr = movavg(ss, 'square-root', 20);
% SSmm_e = movavg(ss, 'exponential', 20);
% 
% 
% window = 30;
% for i=window/2+11:length(ss)-window/2
% %     ss_mm(i) = mean(ss(i-window/2:i+window/2));
%     ss_mm(i) = sqrt(mean(ss(i-window/2:i+window/2).^2));
% %     ss_mm(i) = mean(ss(i-window/2:i+window/2).^6).^(1/6);
% end
% 

% x = linspace(0, 100, length(ss_mm));
% 
% figure
% hold on
% plot(x, ss(window/2+1:end))
% plot(x, ss_mm)
% xlim([59 62])
% hold off
% 
% 
% XX2 = [XX(16:end,:) ss_mm'];

%%

% Xt_Pocos_PCA_06_12_50 = XX;
% Xt_Pocos_PCA_20_40_50 = XX;
Xt_Pocos_PCA_50_100_50mm = XX2;

