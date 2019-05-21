clear
figure(1)

coverage_range = 1.75:0.25:3;
centrilized_Fairness_index = [0.581054106022	
    0.65651305524 
    0.672041960834	
    0.759739786413	
    0.783488957742	
    0.805884761636];
mtsp_Fairness_index = [0.8884 0.9236 0.9473 0.9485 0.9547 0.9736];
distributed_Fairness_index = [0.6854484990326042	
    0.7503319587856074
0.7550228
0.856090184
	0.8819409588613634
	0.920769
];
greedy_Fairness_index=[0.4822355672545717
	0.5800804885672445
	0.6494243152115757
	0.7000459601657465
	0.7677439240538421
 0.7956994163057675


];
ramdom_Fairness_index=[0.4126646929436062
	0.49922170790272147
	0.5902344510202139
	0.644733140516601
	0.7018925760828375
	0.7549240788405139
];
plot(coverage_range,distributed_Fairness_index,'r-*',coverage_range,mtsp_Fairness_index,'m-.^',coverage_range,centrilized_Fairness_index,'g->',coverage_range,greedy_Fairness_index,'b-o',coverage_range,ramdom_Fairness_index,'k-s','Linewidth',2.5,'markersize',10)

xlim([1.6,3.1])
ylim([0.35,1])

set(gca,'xtick', (1.5:0.25:3.25),'fontsize',20)
set(gca,'ytick', (0.3:0.1:1),'fontsize',20)

legend({'Our approach','mTSP','DRL-EC^3','Greedy','Random'},'location','SouthEast','fontsize',13)

xlabel('Coverage range (units)','fontsize',20)
ylabel('Fairness index','fontsize',20)
grid on;
saveas(gcf,'cov_fair.pdf')
