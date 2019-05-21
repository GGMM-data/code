clear
figure(1)

coverage_range = 1.75:0.25:3;
centrilized_Energy_efficiency = [0.493156569506	0.667944101395 0.78897938992	0.988416592737	1.14083710577	1.23169859449];
mtsp_Energy_efficiency = [0.3519 0.4787 0.6012 0.6417 0.6539 0.7698];
distributed_Energy_efficiency = [0.5756288551859164	
    0.7952396550707543	
    0.9221848
    1.1231932206
    1.3462004288420828	
    1.435410
];
greedy_Energy_efficiency=[0.41424010716436394
	0.5994643999628746	
    0.7501135616265492
	0.8819750581523845
	1.0647694247847037
    1.1460801566620855
];
ramdom_Energy_efficiency=[0.3089475407001164
	0.45319349321968994
	0.6374742723723835
	0.7668175275046706
	0.9098665044258581
	1.0596660836098
];
plot(coverage_range,distributed_Energy_efficiency,'r-*',coverage_range,mtsp_Energy_efficiency,'m-.^',coverage_range,centrilized_Energy_efficiency,'g->',coverage_range,greedy_Energy_efficiency,'b-o',coverage_range,ramdom_Energy_efficiency,'k-s','Linewidth',2.5,'markersize',10)

xlim([1.6,3.1])
ylim([0.2,1.5])

set(gca,'xtick', (1.5:0.25:3.25),'fontsize',20)
set(gca,'ytick', (0.2:0.2:1.5),'fontsize',20)

legend({'Our approach','mTSP','DRL-EC^3','Greedy','Random'},'location','NorthWest','fontsize',13)

xlabel('Coverage range (units)','fontsize',20)
ylabel('Energy efficiency','fontsize',20)
grid on;
saveas(gcf,'cov_effi.pdf')
