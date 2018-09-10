# -*- coding: utf-8 -*-
"""
Created on Fri May  4 12:08:10 2018

@author: avarfolomeev
"""
import numpy as np



class ROS_ts:
    "ROS timestamp with UNIX epoch seconds and nanoseconds"
    _billion = 1000000000
    
    def __init__(self, s=0, ns=None):
        if ( s is None):
            self.ns = self._billion - 1
            self.s  = self._billion - 1
        else:    
            _type = type(s)
            if (_type is list or _type is np.ndarray or _type is tuple):        
                self.s = int(s[0]);
                if ( len(s) > 1):
                    self.ns = int(s[1]);
                else:
                    self.ns = 0;
            elif (_type is ROS_ts):
                self.s = s.s;
                self.ns = s.ns;
            else:
                self.s = int(s)
                if (_type == float or _type == np.float32 or _type == np.float64 ):
                    self.ns = int((s-int(s))*self._billion)
                elif (ns is None):                
                    self.ns = int(s);
                    self.s = 0;
                else:
                    self.ns = ns;
            if (self.ns >= self._billion):
                self.s += self.ns // self._billion;
                self.ns = self.ns % self._billion;
        
    def to_string(self):
        return "{:d}.{:09d}".format(self.s, self.ns)
        
    def __lt__(self, right):
        if (type(right) is not ROS_ts):
            right = ROS_ts(right);
        if (self.s == right.s):
            return (self.ns < right.ns);
        else:
            return (self.s < right.s);

    def __le__(self, right):
        if (type(right) is not ROS_ts):
            right = ROS_ts(right);
        if (self.s == right.s):
            return (self.ns <= right.ns);
        else:
            return (self.s < right.s);

    def __eq__(self, right):
        if (type(right) is not ROS_ts):
            right = ROS_ts(right);
        if (self.s == right.s):
            return (self.ns == right.ns);
        else:
            return False;

    def __iadd__ (self,right):
        if (type(right) is not ROS_ts):
            right = ROS_ts(right);
        self.s += right.s;
        self.ns += right.ns;
        if (self.ns >= self._billion):
            self.s += 1;
            self.ns -= self._billion;
        return self    

    def __isub__ (self,right):
        if (type(right) is not ROS_ts):
            right = ROS_ts(right);
        self.s -= right.s;
        self.ns -= right.ns;
        if (self.ns < 0 and self.s > 0):
            self.s -= 1;
            self.ns += self._billion;
        return self

    def __add__ (self,right):
        if (type(right) is not ROS_ts):
            right = ROS_ts(right);
        s = self.s + right.s;
        ns = self.ns + right.ns;
        while (ns < 0):
            ns += self._billion;
            s -= 1;
        if (ns >= self._billion):
            s += 1;
            ns -= self._billion;
        return ROS_ts(s,ns)
            
    #todo: deal with negative numbers!!!
    def __sub__ (self,right):
        if (type(right) is not ROS_ts):
            right = ROS_ts(right);
        s= self.s - right.s;
        ns = self.ns - right.ns;
        if (s < 0 and ns > 0):
            s += 1;
            ns -= self._billion;
        elif (ns < 0 and s > 0):
            s -= 1;
            ns += self._billion;
        return ROS_ts(s,ns)
                    
    def __str__(self):
        ns = self.ns;
        sign = '';
        if (self.s == 0 and ns < 0):
            sign = '-';
            ns = -ns;
        return "{:s}{:d}.{:09d}".format(sign,self.s, ns)

    def __repr__(self):
        ns = self.ns;
        sign = '';
        if (self.s == 0 and ns < 0):
            sign = '-';
            ns = -ns;
        return "({:s}{:d}.{:09d})".format(sign,self.s, ns)
        
    def __copy__(self):
        result = ROS_ts(self.s, self.ns)
        return result    
        
    def double(self):
        result = np.double(self.s) + np.double(self.ns)/1e9
        return result;