clear
figure(1)

coverage_range = 1.75:0.25:3;

centrilized_Normalized_average_energy_consumption =[0.52308542554	
    0.516772683823 
    0.508327460466	
    0.518291543207	
    0.507193897001	
    0.507330639454];
mtsp_Normalized_average_energy_consumption = [1 1 1 1 1 1];
distributed_Normalized_average_energy_consumption = [0.5619371133449522	
    0.5626155703933872
    0.555586926
    0.569874212
    0.5514894662664033
	0.563452
    ];
greedy_Normalized_average_energy_consumption=[0.5500000000000052
	0.5500000000000052
	0.5500000000000052
	0.5500000000000052
	0.5500000000000052	
    0.5500000000000052
    ];
ramdom_Normalized_average_energy_consumptio =[0.5382309731761187
	0.5382848217243701
	0.5382196448908855
	0.5382652940300825
	0.5382547457224585
	0.538270131490073
    ];
plot(coverage_range,distributed_Normalized_average_energy_consumption,'r-*',coverage_range,mtsp_Normalized_average_energy_consumption,'m-.^',coverage_range,centrilized_Normalized_average_energy_consumption,'g->',coverage_range,greedy_Normalized_average_energy_consumption,'b-o',coverage_range,ramdom_Normalized_average_energy_consumptio,'k-s','Linewidth',2.5,'markersize',10)

xlim([1.6,3.1])
ylim([0.5,1.05])

set(gca,'xtick', (1.5:0.25:3.25),'fontsize',20)
set(gca,'ytick', (0.5:0.1:1.05),'fontsize',20)

legend({'Our approach','mTSP','DRL-EC^3','Greedy','Random'},'location','NorthWest','fontsize',13)

xlabel('Coverage range (units)','fontsize',20)
%ylabel({'Normalized Average Energy Consumption'},'fontsize',20)
ylabel({'Average energy consumption'},'fontsize',20)
grid on;
saveas(gcf,'cov_energy.pdf')
