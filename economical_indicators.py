from array import array
from numpy import ndarray
from numpy import array as narray
from statistics import mean, pstdev
from mad import MAD # this can be any function that calculates mean absolute deviation (even a custom script as in this case)

class indicators:
    """
    Class for complex indicator objects.
    """
    def __init__(self,start_date:str, end_date1:str,end_date2:str,date:list|tuple|ndarray|array,price:list|tuple|ndarray|array,index:list|tuple|ndarray|array): # annualised is not part of the class attributes
        """
        Function that calculates some common economical indicators of price data.
        :param start_date: start of period
        :param end_date1: end of period 1
        :param end_date2: end of period 2
        :param date: sequence of DATE data (ordered from oldest to newest)
        :param price: sequence of PRICE data (ordered from oldest to newest)
        :param index: sequence of INDEX data (ordered from oldest to newest)
        """

        # checking given parameters
        if any(type(i)!=str for i in (start_date,end_date1,end_date2)):
            raise TypeError("At least one given parameter, that has to be str, is not!")

        if any(type(i) not in (list,tuple,ndarray,array) for i in (date,price,index)):
            raise TypeError("At least one given parameter, that has to be sequence, is not!")
        
        
        # checking whether dates are contained in date sequence
        if any(i not in date for i in (start_date,end_date1,end_date2)):
            raise Exception("At least one given date is not in date sequence!")
        
        # checking whether sequences have the same length
        if not len(date)==len(price)==len(index):
            raise Exception("Given sequences have different length!")

        # calculating period lengths
        for i in range(0, len(date)):
            if date[i]==start_date:
                break
            if date[i] == end_date1:
                length1 = i
            if date[i] == end_date2:
                length2 = i

        length1=i-length1
        length2=i-length2

        
        #SMA, CCI, STOCHASTIC, BB, TD, ROC - shouldwork
        sma = narray([0] * (len(price) - length1), dtype=float)
        cci = narray([0] * (len(price) - length1), dtype=float)
        stochastic = narray([0] * (len(price) - length1), dtype=float)
        bb = narray([0] * (len(price) - length1), dtype=float)
        td = narray([0] * (len(price) - length1), dtype=float)
        roc = narray([0] * (len(price) - length1), dtype=float)

        #RSI
        avg_gain = narray([0] * (len(price) - length1), dtype=float) # is the length correct here? - I believe it is
        avg_loss = narray([0] * (len(price) - length1), dtype=float)
        rsi = narray([0] * (len(price) - length1), dtype=float)

        #AROON
        aroon_up = narray([0] * (len(price) - length1), dtype=float)
        aroon_down = narray([0] * (len(price) - length1), dtype=float)

        for i in range(length1, len(price)):
            #SMA
            sma[i-length1] = mean(price[i-length1:i+1]) #current day's data is also used for calculation, which makes it unsuitable for prediction - if needed, current day's data can be ignored

            #CCI
            cci[i - length1] = (price[i] - mean(price[i - length1:i + 1])) / MAD(price[i - length1:i + 1])

            #Stochastic
            stochastic[i-length1] = (price[i] - min(price[i-length1:i + 1])) / (
                    max(price[i-length1:i + 1]) - min(price[i-length1:i + 1])) * 100 if (max(price[
                                                                                                 i-length1:i + 1]) - min(
                price[i-length1:i + 1])) > 0 else 0
            #BB
            bb[i - length1] = (mean(price[i - length1:i + 1]) + 2 * pstdev(price[i - length1:i + 1])) - (
                    mean(price[i - length1:i + 1]) - 2 * pstdev(price[i - length1:i + 1]))
            #TD
            td[i - length1] = (
                    (price[i] - price[i - length1]) / price[i - length1] - (index[i] - index[i - length1]) / index[
                i - length1])

            #ROC
            roc[i - length1] = (price[i] - price[i - length1]) / price[i - length1] * 100

            #RSI
            if i == length1:
                pal = 0
                pag = 0
                for j in range(1, length1 + 1):  # i+1->i+length1-1
                    if price[j] - price[j - 1] < 0: # if the price changed downwards considering previous day
                        pal = pal + (abs(price[j] - price[j - 1]))
                    if price[j] - price[
                        j - 1] > 0:  # if the price changed upwards considering previous day
                        pag = pag + (price[j] - price[
                            j - 1])  # len(pal)+len(pag)==length1-1 -> can be False, since  if no change in price occured, we do no not put the 'difference' in any of the lists
                pal = pal / (length1 + 1)  #average loss during the period
                pag = pag / (length1 + 1)  #average gain during the period
                avg_loss[i - length1] = pal 
                avg_gain[i - length1] = pag 
                rsi[i - length1] = 100 - 100 / (1 + avg_gain[i - length1] / avg_loss[i - length1])
            else:
                avg_loss[i - length1] = (avg_loss[i - length1 - 1] * length1 + (
                    0 if price[i] - price[i - 1] > 0 else abs(price[i] - price[i - 1]))) / (length1 + 1)
                avg_gain[i - length1] = (avg_gain[i - length1 - 1] * length1 + (
                    0 if price[i] - price[i - 1] < 0 else abs(price[i] - price[i - 1]))) / (length1 + 1)
                rsi[i - length1] = 100 - 100 / (1 + avg_gain[i - length1] / avg_loss[i - length1])

            #AROON
            aroon_up[i - length1] = ((length1 + 1) - (
                        price[i - length1:i + 1].index(max(price[i - length1:i + 1])) + 1)) / (length1 + 1)
            aroon_down[i - length1] = ((length1 + 1) - (
                        price[i - length1:i + 1].index(min(price[i - length1:i + 1])) + 1)) / (
                                                 length1 + 1)  # since the current value has to be included:  length1+1

        self.sma=sma
        self.cci = cci
        self.stochastic = stochastic
        self.bb = bb
        self.td = td
        self.roc=roc
        self.rsi = rsi
        aroon = (aroon_up - aroon_down) * 100
        self.aroon = aroon

        #KAMA - Source: https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/kaufmans-adaptive-moving-average-kama  shouldwork
        period = 10 # they use this with a standard value of 10
        fastest = 2 / 3
        slowest = 2 / 31

        kama = narray([0] * (len(price) - (period-1)), dtype=float)
        kama[0] = mean(price[0:period]) # first KAMA data point is the mean of previous prices

        for i in range(period, len(price)):
            change=abs(price[i]-price[i-period]) # change in absolute terms between price today and the price 10 days ago
            volatility = 0 # fluctuation of the price in the last 10 day period
            for j in range(i - period+1, i + 1):
                volatility = volatility + abs(price[j] - price[
                    j - 1])  # summing up the daily absolute change in price during the last 10 day period
            er = change / volatility if volatility > 0 else 0  # avoiding ZeroDivisionError
            sc = (er * (fastest - slowest) + slowest) ** 2  # calculating the smoothing constant

            kama[i - period+1] = kama[i - period] + sc * (price[i] - kama[
                i - period])  # calculating today's KAMA

        self.kama=kama
        
        #PPO
        ppo = narray([0] * (len(price) - length2), dtype=float)
        for i in range(length2, len(price)):
            ppo[i - length2] = (mean(price[i - length1:i + 1]) - mean(price[i - length2:i + 1])) / mean(
                price[i - length2:i + 1]) * 100  # *100 can be abandoned
        self.ppo=ppo

        #MD
        md = []
        for i in range(1, len(price)):
            if price[i] > max(price[0:i]):
                md.append((min(price[price.index(max(price[0:i])):i]) - max(price[0:i])) / max(price[0:i]))
        self.md=min(md)*100


def ANNUALISED(nmbr_of_yrs,price,output_wish, file_):
    if not isinstance(nmbr_of_yrs, (int,float)):
        raise TypeError("Parameter nmbr_of_yrs has to be a number!")
    if not isinstance(price, (list, tuple, ndarray,array)):
        raise TypeError(" Parameter price has to be sequence!")
    if not isinstance(output_wish, bool):
        raise TypeError("Parameter output_wish has to be a number!")
    if not isinstance(file_, str):
        raise TypeError("Parameter file_ has to be a number!")
    if output_wish==True:
        with open(file=file_,mode="w", encoding="utf-8") as f:
            f.write(f"{((price[-1]/price[0])**(1/nmbr_of_yrs)-1)*100}")
    return ((price[-1]/price[0])**(1/nmbr_of_yrs)-1)*100



