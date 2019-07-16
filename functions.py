#!/usr/bin/env python
# coding: utf-8

# In[2]:


def getInterval(fps):
    interval =1.0/fps
    time_wait = (int)(interval * 1000.0)
    print('interval: ',interval,'time_wait: ',time_wait)
    return interval,time_wait

# format time(second) as HH:MM:ss.mmm
def format_time(sec):
    msec = (int)(sec * 1000.)
    s, ms = divmod(msec, 1000)
    m, s =divmod(s,60)
    return '{:02d}:{:02d}.{:03d}'.format(m,s,ms)

def format_time_forfile(sec):
    msec = (int)(sec * 1000.)
    s, ms = divmod(msec, 1000)
    m, s =divmod(s,60)
    return '{:02d}m{:02d}s{:03d}ms'.format(m,s,ms)


# In[ ]:





# In[ ]:




