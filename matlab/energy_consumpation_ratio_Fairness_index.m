clear
figure(1)

energy_consumption_ratio = 0.25:0.5:2.75;

%centrilized_Fairness_index = [0.561	0.803860056516	0.830444828112	0.805884761636	0.811116304201	0.695491690376];
%centrilized_Fairness_index = [0.561	0.803860056516	0.830444828112	0.805884761636	0.811116304201	0.7873024];
centrilized_Fairness_index = [0.76	0.803860056516	0.830444828112	0.805884761636	0.811116304201	0.7873024];
mtsp_Fairness_index = [0.9736 0.9736 0.9736 0.9736 0.9736 0.9736];
distributed_Fairness_index = [ 0.919667834032	0.903533801994	0.94517843309	0.920769	0.950831	0.931647296632];
greedy_Fairness_index=[ 0.807013062441
	0.811979589468
	0.818151166322
	0.7956994163057675
	0.824212468309
	0.828747383372
];
ramdom_Fairness_index=[0.743832346172
	0.762631372428
	 0.741455797507
	0.7549240788405139
	0.775990544432
	0.773098954333
];
plot(energy_consumption_ratio,distributed_Fairness_index,'r-*',energy_consumption_ratio,mtsp_Fairness_index ,'m-.^',energy_consumption_ratio,centrilized_Fairness_index,'g->',energy_consumption_ratio,greedy_Fairness_index,'b-o',energy_consumption_ratio,ramdom_Fairness_index,'k-s','Linewidth',2.5,'markersize',10)

xlim([0,3])
ylim([0.7,1.2])

set(gca,'fontsize',20)
set(gca,'xtick',(0.25:0.5:2.75),'ytick', (0.7:0.1:1.2))
set(gca,'xticklabel',{'0.2','0.3','0.4','0.5','0.6','0.7'})


legend({'Our approach','mTSP','DRL-EC^3','Greedy','Random'},'location','NorthWest','fontsize',13)

xlabel({'Normalized energy consumption' ,'for flying with max distance (unit)'},'fontsize',20)
ylabel('Fairness index','fontsize',20)
grid on;
saveas(gcf,'ratio_fair.pdf')
