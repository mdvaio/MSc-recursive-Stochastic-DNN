X = [Xt_PocoA; Xt_PocoB_run1; Xt_PocoB_run2]
SS = [Xt_PocoA_SS; Xt_PocoB_run1_SS; Xt_PocoB_run2_SS]




Y = tsne(X);


SS1 = Xt_PocoA_SS;
SS2 = Xt_PocoB_run1_SS;
SS3 = Xt_PocoB_run2_SS;
dimensoes = [20127 length(SS1)-20127 length(SS2) length(SS3)];
size_pocos = cumsum(dimensoes);


a = linspace(1, length(Y), length(Y));
Y2 = [Y a'];


figure
count = 0; count_ant = 1;
for i=1:length(size_pocos)
    
    count = size_pocos(i);
   
    Proj1 = Y2(count_ant:count,1);
    Proj2 = Y2(count_ant:count,2);
    Proj3 = Y2(count_ant:count,3);
    
    scatter3(Proj1,Proj2,Proj3, [20], SS(count_ant:count), 'filled')
    hold on
    count_ant = count;
end

hold off
colorbar('Ticks', [0.5, 0.8 1.2 2.25 3.5]); caxis([0.5 2.7])
%xlabel('RPM'); ylabel('SWOB'); zlabel('STOR')
colormap(jet);
grid on