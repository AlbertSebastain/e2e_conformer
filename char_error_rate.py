import Levenshtein
class char_error:
    def __init__(self,rescore):
        self.spkid_er_rate = {}
        self.spkid_er_sum = {}
        self.uttid_er_rate = {}
        #self.spkid_num = {}
        self.rescore = rescore
        self.n = 0
        self.error = []
    def error_rate_compute(self,text1,text2,spkid,uttid):
        text2 = text2[0:-1]
        str1 = "".join(text1)
        str2 = "".join(text2)
        self.error.append(Levenshtein.ratio(str1,str2))
        self.n += 1
        if self.n == self.rescore:
            self.n = 0
            #self.error = []
            error_minmum = min(self.error)
            self.uttid_er_rate[uttid] = error_minmum
            if spkid in self.spkid_er_rate.keys():
                #self.spkid_num[spkid] = self.spkid_num[spkid]+1
                self.spkid_er_sum[spkid].append(error_minmum)
                self.spkid_er_rate[spkid] = mean(self.spkid_er_sum[spkid])
            else:
                self.spkid_er_sum[spkid] = [error_minmum]
                self.spkid_er_rate[spkid] = error_minmum
            self.error = []
        return 

        

text1 = u"加油"
text2 = u"家有"
dist = Levenshtein.ratio(text1,text2)
print(dist)