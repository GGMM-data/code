clear
figure(2)

uav_numbers = 3:1:8;

centrilized_fairness_index = [0.596128633718	0.700620283794	0.784764731579	0.805884761636	0.858117758712	0.869017855324];
mtsp_fairness_index = [0.9017 0.9347 0.9650 0.9736 0.9821 0.9869];
distributed_fairness_index = [0.7104272407195077	
    0.7502566252864014	
    0.809622505358996	
    0.920769	
    0.922196671718671	
    0.949885];
greedy_fairness_index=[0.6290897659032756
	0.7165380110099242
	0.7697500138070076
	 0.7956994163057675
	0.8425626601713262
	0.885245602285815
];
ramdom_fairness_index=[0.5199648837987856
	 0.6293682918852597
	 0.7103678385153139
	0.7564170578060109
	0.8117398858786165
	0.8451726212774452

];
plot(uav_numbers,distributed_fairness_index,'r-*',uav_numbers,mtsp_fairness_index ,'m-.^',uav_numbers,centrilized_fairness_index,'g->',uav_numbers,greedy_fairness_index,'b-o',uav_numbers,ramdom_fairness_index,'k-s','Linewidth',2.5,'markersize',10)

xlim([2.7,8.3])
ylim([0.5,1])

set(gca,'xtick', (3:1:8),'fontsize',20)
set(gca,'ytick', (0.5:0.1:1),'fontsize',20)

legend({'Our approach','mTSP','DRL-EC^3','Greedy','Random'},'location','SouthEast','fontsize',13)

xlabel('Number of UAVs','fontsize',20)
ylabel('Fairness index','fontsize',20)
grid on;
saveas(gcf,'UAV_fair.pdf')
