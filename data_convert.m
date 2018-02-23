function data = data_convert(dat)
formatIn = 'mm/dd/yyyy';
index = 1; % the subject number
Idx(1) = dat.id(2);
t = 1;
i = 2;
    age(index) = dat.age(i);
    edu(index) = dat.edu(i);
    time(index,t) = datenum(dat.date{i},formatIn);
    T(index,t)= (time(index,t)-time(index,1))/365 + age(index);
    
    Y(1,t,index) = dat.MMSE(i);
    Y(2,t,index) = dat.ADAS(i);
    Y(3,t,index) = dat.ven(i)/dat.brain(i);
    Y(4,t,index) = dat.hip(i)/dat.brain(i);
    Y(5,t,index) = dat.cdrs(i);
    label{index,t} = dat.label{i};
    change(index) = dat.change(i);
  
    X(index,1) = dat.APOE(i);
    if length(dat.gender{i}) == 4
      X(index,2) = 1;
    else
      X(index,2) = 0;
    end
    X(index,3) = dat.edu(i);
display(Idx)
for i = 3:length(dat.date)
    if i == 3505
        continue;
    end
    if dat.id(i)==Idx(index) 
    t = t + 1;
    X(index,1) = dat.APOE(i);
    display(i)    
    if length(dat.gender{i}) == 4
      X(index,2) = 1;
    else
      X(index,2) = 0;
    end
    X(index,3) = dat.edu(i);
    age(index) = dat.age(i);
    edu(index) = dat.edu(i);
    time(index,t) = datenum(dat.date{i},formatIn);
    T(index,t)= (time(index,t)-time(index,1))/365 + age(index);
  
    Y(1,t,index) = dat.MMSE(i);
    Y(2,t,index) = dat.ADAS(i);
    Y(3,t,index) = dat.ven(i)/dat.brain(i);
    Y(4,t,index) = dat.hip(i)/dat.brain(i);
    Y(5,t,index) = dat.cdrs(i);
    label{index,t} = dat.label{i};
    change(index) = dat.change(i);
    
    else  
    t = 1;       
    index = index+1;        
    Idx(index) = dat.id(i);
    age(index) = dat.age(i);
    edu(index) = dat.edu(i);
    time(index,t) = datenum(dat.date{i},formatIn);
    T(index,t)= (time(index,t)-time(index,1))/365 + age(index);
    Y(1,t,index) = dat.MMSE(i);
    Y(2,t,index) = dat.ADAS(i);
    Y(3,t,index) = dat.ven(i)/dat.brain(i);
    Y(4,t,index) = dat.hip(i)/dat.brain(i);
    Y(5,t,index) = dat.cdrs(i);
    label{index,t} = dat.label{i};
    change(index) = dat.change(i);
      X(index,1) = dat.APOE(i);
    if length(dat.gender{i}) == 4
      X(index,2) = 1;
    else
      X(index,2) = 0;
    end
      X(index,3) = dat.edu(i);
    end   
end


Y(isnan(Y)) = 0;
X(isnan(X)) = 0;
data.X = X';
data.age = age';
data.time = time';
data.Y = Y;
data.idx = Idx';
data.label = label';
data.T = T';
data.change = change;
data.edu = edu;

