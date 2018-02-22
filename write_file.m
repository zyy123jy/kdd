function write_file(data)
X=data.X;
Y=data.Y;
T=data.T;
fid = fopen('X.txt','w');
for i=1:length(X)
    fprintf(fid,'%5d\t%5d\t%5d\n',X(:,i));
    fprintf('\n');
end
fclose(fid)

fid = fopen('T.txt','w');
for i=1:length(T)
    t=T(:,i);
    for j=1:length(t)
    fprintf(fid,'%3.3f\t',t(j));
    end
    fprintf(fid,'\n');
end
fclose(fid)

fid = fopen('Y1.txt','w');
for i=1:length(Y)
    y=Y(1,:,i);
    for j=1:length(y)
    fprintf(fid,'%5d\t',y(j));
    end
    fprintf(fid,'\n');
end
fclose(fid)

fid = fopen('Y2.txt','w');
for i=1:length(Y)
    y=Y(2,:,i);
    for j=1:length(y)
    fprintf(fid,'%2.2f\t',y(j));
    end
    fprintf(fid,'\n');
end
fclose(fid)  

fid = fopen('Y3.txt','w');
for i=1:length(Y)
    y=Y(3,:,i);
    for j=1:length(y)
    fprintf(fid,'%2.4f\t',y(j));
    end
    fprintf(fid,'\n');
end
fclose(fid)

fid = fopen('Y4.txt','w');
for i=1:length(Y)
    y=Y(4,:,i);
    for j=1:length(y)
    fprintf(fid,'%2.4f\t',y(j));
    end
    fprintf(fid,'\n');
end
fclose(fid)  

fid = fopen('Y5.txt','w');
for i=1:length(Y)
y=Y(5,:,i);
for j=1:length(y)
fprintf(fid,'%2.4f\t',y(j));
end
fprintf(fid,'\n');
end

