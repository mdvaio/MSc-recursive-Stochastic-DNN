

% Cria Matriz unida e exporta para txt




Xt_PocosB_06_12_50 = [Xt_PocoBr1_06_12_50; Xt_PocoBr2_06_12_50];
Xt_PocosB_20_40_50 = [Xt_PocoBr1_20_40_50; Xt_PocoBr2_20_40_50];
Xt_PocosB_50_100_50 = [Xt_PocoBr1_50_100_50; Xt_PocoBr2_50_100_50];





dlmwrite('Xt_Pocos_PCA_06_12_20.txt', Xt_Pocos_PCA_06_12_50, 'delimiter','\t','newline','pc')
dlmwrite('Xt_Pocos_PCA_20_40_50.txt', Xt_Pocos_PCA_20_40_50, 'delimiter','\t','newline','pc')
dlmwrite('Xt_Pocos_PCA_50_100_50.txt', Xt_Pocos_PCA_50_100_50, 'delimiter','\t','newline','pc')

dlmwrite('Xt_PocosB_06_12_20.txt', Xt_PocosB_06_12_50, 'delimiter','\t','newline','pc')
dlmwrite('Xt_PocosB_20_40_50.txt', Xt_PocosB_20_40_50, 'delimiter','\t','newline','pc')
dlmwrite('Xt_PocosB_50_100_50.txt', Xt_PocosB_50_100_50, 'delimiter','\t','newline','pc')


dlmwrite('Xt_PocoA_06_12_20.txt', Xt_PocoA_06_12_50, 'delimiter','\t','newline','pc')
dlmwrite('Xt_PocoA_20_40_50.txt', Xt_PocoA_20_40_50, 'delimiter','\t','newline','pc')
dlmwrite('Xt_PocoA_50_100_50.txt', Xt_PocoA_50_100_50, 'delimiter','\t','newline','pc')



dlmwrite('Xt_Pocos_PCA_50_30_50.txt', Xt_Pocos_PCA_50_30_50, 'delimiter','\t','newline','pc')
dlmwrite('Xt_Pocos_PCA_NEW_50_30_50.txt', Xt_Pocos_PCA_NEW_50_30_50, 'delimiter','\t','newline','pc')
dlmwrite('Xt_Pocos_PCA_NEWNORM_50_30_50.txt', Xt_Pocos_PCA_NEWNORM_50_30_50, 'delimiter','\t','newline','pc')
dlmwrite('Xt_Pocos_PCA_NEWNORM_ssmm_50_30_50.txt', Xt_Pocos_PCA_NEWNORM_ssmm_50_30_50, 'delimiter','\t','newline','pc')


%%

dlmwrite('Bit_Speed.txt', Bit_Speed_Sample_175_complete, 'delimiter','\t','newline','pc')
dlmwrite('Bit_Torque.txt', Bit_Torque_Sample_175_complete, 'delimiter','\t','newline','pc')
dlmwrite('Time_Sample.txt', Time_Sample_175_complete, 'delimiter','\t','newline','pc')
dlmwrite('Top_Speed.txt', Top_Speed_Sample_175_complete, 'delimiter','\t','newline','pc')
dlmwrite('WOB_Sample.txt', WOB_Sample_175_complete, 'delimiter','\t','newline','pc')


