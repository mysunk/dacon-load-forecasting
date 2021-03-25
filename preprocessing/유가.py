import os
import time
import numpy as np
import pandas as pd
from datetime import datetime


def get_oil_price(code):
    
    """
        Description
        -----------
        유가 데이터 수집 프로그램(from 네이버)
        
        Input
        -----
        code : OIL_CL(WTI), OIL_DU(두바이유), OIL_LO(경유), ...
        
        Output
        ------
        일별 유가(종가)
        
        Example
        -------
        oil_price_du = get_oil_price('OIL_DU')
        
    """
    
    delay = 0.01
    page = 4
    result = []
    start_time = datetime.now()
    
    # 수집
    print('[{}] 데이터 수집을 시작합니다. (code: {})'.format(start_time.strftime('%Y/%m/%d %H:%M:%S'), code))
    while(True):
        # url = 'https://finance.naver.com/marketindex/worldDailyQuote.nhn?marketindexCd={}&fdtc=2&page={}'.format(code, page)
        url = 'https://finance.naver.com/marketindex/exchangeDetail.nhn?marketindexCd={}&page={}'.format(code, page)
        # data = pd.read_html(url)[0].dropna()
        res = req.urlopen(url)
        soup = BeautifulSoup(res, "html.parser")
        
        if page != 1:
            try:
                if data.iloc[-1, 0] == result[-1].iloc[-1, 0]:
                    break
            except:
                break
        result.append(data)
        page += 1
        time.sleep(delay)
    
    # 가공
    oil_price = pd.concat(result).reset_index(drop=True)
    # oil_price.columns = ['날짜', '종가', '전일대비', '등락율']
    oil_price.columns = ['날짜', '매매기준율', '전일대비', '사실때','파실때','보내실때','받으실때']
    oil_price['날짜'] = oil_price['날짜'].apply(lambda x: datetime.strptime(x, '%Y.%m.%d'))
    oil_price = oil_price[['날짜', '종가']]
    oil_price.insert(0, '코드', code)
    
    end_time = datetime.now()
    print('[{}] 데이터 수집을 종료합니다. (code: {}, 수집시간: {}초, 데이터수: {:,}개)'.format(end_time.strftime('%Y/%m/%d %H:%M:%S'), code, (end_time-start_time).seconds, len(oil_price)))
    return oil_price

# 예시
LNG = get_oil_price('FX_USD')
