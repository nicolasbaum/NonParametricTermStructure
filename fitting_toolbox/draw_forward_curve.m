function draw_forward_curve(rr)
    global glob;
    time = glob.tspan;
    tmp = rr.*time;
    % figure;
    hold on;
    plot(time(1:end-1),diff(tmp)*100./diff(time),'b-');
end

