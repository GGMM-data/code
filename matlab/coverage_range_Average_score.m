clear
figure(1)

coverage_range = 1.75:0.25:3;
centrilized_average_score = [0.444395	0.5251021 0.5941791	0.6680962	0.7336291	0.7654405];
mtsp_average_score = [0.3954 0.5173 0.6346 0.6771 0.6844 0.7904];
distributed_average_score = [
    0.4718818	
    0.5960592
    0.6765214
    0.7475914
    0.8415920
	0.878381
];
greedy_average_score=[0.4493273999999999
	0.562437
	0.6263476
	0.6826548000000001
	0.7517704
    0.7806045999999999
];
ramdom_average_score=[0.39542980000000005
	0.4816898
	0.5737418000000001
	0.6299956
	0.688991
	0.7445887999999999
];
plot(coverage_range,distributed_average_score,'r-*',coverage_range, mtsp_average_score, 'm-.^', coverage_range,centrilized_average_score,'g->',coverage_range,greedy_average_score,'b-o',coverage_range,ramdom_average_score,'k-s','Linewidth',2.5,'markersize',10)

xlim([1.6,3.1])
ylim([0.3,1])

set(gca,'xtick', (1.5:0.25:3.25),'fontsize',20)
set(gca,'ytick', (0.3:0.1:1),'fontsize',20)

legend({'Our approach','mTSP','DRL-EC^3','Greedy','Random'},'location','NorthWest','fontsize',13)

xlabel('Coverage range (units)','fontsize',20)
ylabel('Average coverage score','fontsize',20)
grid on;
saveas(gcf,'cov_score.pdf')

