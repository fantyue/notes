SELECT  distinct 
    d1.uid AS seller_id,
    nvl(d16.line_level,'其他') AS city_level,
    CASE WHEN d4.uid is null THEN 0  ELSE 1 END AS publish_year,
    CASE WHEN d41.uid is null THEN 0  ELSE 1 END AS publish_phone_pad_year,
    CASE WHEN d42.uid is null THEN 0  ELSE 1 END AS publish_phone_pad_30days,
    CASE WHEN d3.seller_order>0 THEN 1  ELSE 0 END AS sell_year,
    CASE WHEN d3.buyer_order>0 THEN 1  ELSE 0 END AS buy_year,
    CASE WHEN d3.seller_phone_order>0 THEN 1  ELSE 0 END AS sell_phone_year,
    CASE WHEN d3.buyer_phone_order>0 THEN 1  ELSE 0 END AS buy_phone_year,
    if(d5.uid is null,0,1) AS user_type_service_create,
    if(d6.seller_id is null,0,1) AS bm_isdeal,
    if(d3.bm_seller_order>0,1,0) AS user_type_service_deal,
    CASE WHEN d7.buyer_id is null THEN 0  ELSE 1 END  AS buy_month,
    if(d8.uid is null,0,1) AS click_deal,
    if(d9.uid is null,0,1) AS click_kf,
    CASE
        WHEN DATEDIFF(d1.dt,from_unixtime(cast(d11.timestamp/1000 AS BIGINT),'yyyy-MM-dd'))>10 
        THEN 1  ELSE 0 END AS register,
    if(d12.createOrder is null,0,1) AS is_createorder,
    if(d12.delivery is null,0,1) AS is_delivery,if(d13.token is null,0,1) AS is_eval_30days,
    if(d14.token is null,0,1) AS active,
    if(d15.token is null,0,1) AS same_modle_3days
FROM 
( --半年活跃用户
	select distinct
        token,
	    uid
    from
    (select 
        token,
        dt
    from hdp_zhuanzhuan_dw_global.dw_user_label_token_full_1d
    where dt="${outFileSuffix}" and label_name="biz_biz_hyuser" and label_value = "近30天活跃用户") a
    left join(
    select DISTINCT
	    token,
	    uid
    from hdp_zhuanzhuan_rawdb_global.raw_token
    where dt="${outFileSuffix}" ) b
    on a.token=b.token
) d1
LEFT JOIN hdp_zhuanzhuan_rawdb_global.raw_t_merchant_all_online_1d d2
ON d2.dt='${outFileSuffix}' AND d1.uid = d2.uid
LEFT JOIN hdp_zhuanzhuan_dm_global.dm_bm_usertype_full_0p d3
ON d1.uid = d3.uid
LEFT JOIN 
(--是否发布过
	SELECT  distinct uid
	FROM hdp_zhuanzhuan_dw_global.dw_mysql_info_1d
	WHERE dt='${outFileSuffix}' 
	AND from_unixtime(cast(timestamp/1000 AS BIGINT),'yyyy-MM-dd') BETWEEN date_sub('${outFileSuffix}',365) AND '${outFileSuffix}'  
) d4
ON d4.uid = d1.uid
LEFT JOIN 
(--是否发布过手机平板
	SELECT  distinct uid
	FROM hdp_zhuanzhuan_dw_global.dw_mysql_info_1d
	WHERE dt='${outFileSuffix}' 
	AND from_unixtime(cast(timestamp/1000 AS BIGINT),'yyyy-MM-dd') BETWEEN date_sub('${outFileSuffix}',365) AND '${outFileSuffix}' 
	AND cate_first_id in(101,109)  
) d41
ON d41.uid = d1.uid
LEFT JOIN 
(--30天内是否发布过手机平板
	SELECT  distinct uid
	FROM hdp_zhuanzhuan_dw_global.dw_mysql_info_1d
	WHERE dt='${outFileSuffix}' 
	AND from_unixtime(cast(timestamp/1000 AS BIGINT),'yyyy-MM-dd') BETWEEN date_sub('${outFileSuffix}',30) AND '${outFileSuffix}' 
	AND cate_first_id in(101,109)  
) d42
ON d42.uid = d1.uid
LEFT JOIN 
(--保卖下单
	SELECT  a.seller_id AS uid
	FROM hdp_zhuanzhuan_rawdb_global.raw_trade_t_recycle_order_full_1d a
	WHERE from_unixtime(cast(a.create_time/1000 AS BIGINT),'yyyy-MM-dd') BETWEEN date_sub('${outFileSuffix}',365) AND date_sub('${outFileSuffix}',1) 
	AND a.dt='${outFileSuffix}' 
	AND a.order_source in(21,22) --保卖全部 
	AND a.del=0 
	GROUP BY  a.seller_id 
) d5
ON d1.uid = d5.uid --and d5.rank=2
LEFT JOIN 
(--保卖是否成交
	SELECT  distinct a.seller_id
	FROM hdp_zhuanzhuan_rawdb_global.raw_trade_t_recycle_order_full_1d a
	WHERE a.dt='${outFileSuffix}' 
	AND state=80 
	AND a.order_source in(17,18,21,22,34,35)  
) d6
ON d6.seller_id = d1.uid
LEFT JOIN 
(---买入行为
	SELECT  distinct t1.buyer_id
	FROM hdp_zhuanzhuan_dw_global.dw_mysql_order_1d t1
	WHERE t1.dt='${outFileSuffix}' 
	AND t1.is_trade_success=1 
	AND t1.cate_first_id=101 
	AND from_unixtime(cast(t1.pay_time/1000 AS BIGINT),'yyyy-MM-dd') BETWEEN date_sub('${outFileSuffix}',365) AND date_sub('${outFileSuffix}',1)  
) d7
ON d1.uid = d7.buyer_id
LEFT JOIN 
(--点击历史成交
	SELECT  a.dt 
	       ,a.uid
	FROM hdp_zhuanzhuan_dw_global.dw_log_lego_action_1d a
	WHERE a.dt='${outFileSuffix}' 
	AND a.region='b' 
	AND a.pagetype='ZHUANZHUANM' 
	AND a.actiontype IN ('BM-SELL-HEADER-GO-HISTORY__CLICK','BM-SELL-LIST-GO-HISTORY__CLICK') --and a.datapool['channel']='BM_00292' 
	GROUP BY  a.dt 
	         ,a.uid 
) d8
ON d8.uid = d1.uid
LEFT JOIN 
(--联系客服
	SELECT  a.dt 
	       ,a.uid
	FROM hdp_zhuanzhuan_dw_global.dw_log_lego_action_1d a
	WHERE a.dt='${outFileSuffix}' 
	AND a.region='b' 
	AND a.pagetype='ZHUANZHUANM' 
	AND a.actiontype = 'BM-CORRELATE-KEFU__CLICK' 
	GROUP BY  a.dt 
	         ,a.uid 
) d9
ON d9.uid = d1.uid
LEFT JOIN 
(--保卖客服回复
	SELECT  distinct substr(a.c8,1,10) AS create_date 
	       ,a.c8                       AS create_time 
	       ,a.c15                      AS uid 
	       ,a.c3                       AS `技能组id`
	FROM hdp_ubu_zhuanzhuan_defaultdb.t_statistics_zzim_conversationdetail_info_back a
	WHERE substr(a.c8,1,10)='${outFileSuffix}' 
	AND a.c3 in('10044','10094') 
	AND a.partition_year = '2021'  
) d10
ON d10.uid = d1.uid --注册时间
LEFT JOIN hdp_zhuanzhuan_dw_global.dw_mysql_user_1d d11
ON d11.uid = d1.uid AND d11.dt = '${outFileSuffix}'
LEFT JOIN 
(--当天是否有发货
	SELECT  DISTINCT a.dt 
	       ,a.uid 
	       ,a.token                                          AS createOrder 
	       ,CASE WHEN b.orderid is not null THEN a.token END AS delivery
	FROM hdp_zhuanzhuan_dw_global.dw_log_server_action_1d a
	LEFT JOIN 
	(
		SELECT  a.uid 
		       ,a.token 
		       ,a.datapool['orderid'] AS orderid
		FROM hdp_zhuanzhuan_dw_global.dw_log_server_action_1d a
		WHERE action='bm_delivery' 
		AND region='b' 
		AND dt='${outFileSuffix}' 
		AND datapool['ordersource'] IN (21,22)  
	) b
	ON a.datapool['orderid']=b.orderid
	WHERE a.action='bm_createOrder' 
	AND a.region='b' 
	AND a.dt='${outFileSuffix}' 
	AND a.datapool['ordersource'] IN (21,22)  
) d12
ON d12.createOrder = d1.token
LEFT JOIN 
(--过去30天是否估价
	SELECT  token
	FROM hdp_ubu_zhuanzhuan_dw_c2b.dw_trade_recycle_eval_success_dtl_inc_1d
	WHERE dt BETWEEN date_sub('${outFileSuffix}',31) AND date_sub('${outFileSuffix}',1) 
	GROUP BY  token 
) d13
ON d13.token = d1.token
LEFT JOIN 
(--转转app新增激活
	SELECT  from_unixtime(unix_timestamp(a.date,'yyyyMMdd'),'yyyy-MM-dd') AS dt 
	       ,token 
	       ,uid --token terminal uid version
	FROM hdp_ubu_zhuanzhuan_defaultdb.t_zhuanzhuan_new_dau a
	WHERE a.date='${dateSuffix}' --from_unixtime（unix_timestamp（a.date 'yyyyMMdd'） 'yyyy-MM-dd'） = '${outFileSuffix}'  
) d14
ON d1.token=d14.token
LEFT JOIN 
(--过去三天是否浏览过同型号商详
	SELECT  g.token
	FROM 
	( --估价成功
		SELECT  token 
		       ,model_id
		FROM hdp_ubu_zhuanzhuan_dw_c2b.dw_trade_recycle_eval_success_dtl_inc_1d a
		WHERE dt ='${outFileSuffix}' 
		AND cate_id IN (101,119) 
		GROUP BY  token 
		         ,model_id 
	) g
	LEFT JOIN 
	(-- 浏览过b2c商详页
		SELECT  token 
		       ,model_id
		FROM hdp_zhuanzhuan_dm_global.dm_trade_visit_detail_1d a --全量订单
		WHERE a.dt BETWEEN date_sub('${outFileSuffix}',3) AND '${outFileSuffix}' 
		AND a.cate_grand_id IN (101,119) 
		GROUP BY  token 
		         ,model_id 
	) b2c
	ON g.token = b2c.token AND g.model_id = b2c.model_id
	WHERE b2c.token is not null 
	GROUP BY  g.token 
) d15
ON d1.token=d15.token
LEFT JOIN 
(--估价用户城市等级
	SELECT  distinct a.dt 
	       ,a.token 
	       ,a.city 
	       ,b.line_level
	FROM 
	(
		SELECT  a.dt 
		       ,a.token 
		       ,a.city 
		       ,ROW_NUMBER() over(PARTITION by token ORDER BY timestamp desc) AS rk
		FROM hdp_zhuanzhuan_dw_global.dw_log_lego_appstart_1d a
		WHERE a.city is not null 
		AND dt='${outFileSuffix}'  
	) a
	LEFT JOIN hdp_ubu_zhuanzhuan_defaultdb.bm_city_map b
	ON a.city=b.city1
	WHERE a.rk=1  
) d16
ON d16.token=d1.token
LEFT JOIN 
( --估价单均价
	SELECT  a.token 
	       ,CEIL(AVG(nvl(highest_price,0))) AS avg_price 
	       ,CEIL(AVG(nvl(coupon_ratio,10))) AS coupon
	FROM 
	(
		SELECT  a.token 
		       ,a.coupon_ratio 
		       ,a.qc_code 
		       ,a.highest_price
		FROM hdp_ubu_zhuanzhuan_dw_c2b.dw_trade_recycle_eval_success_dtl_inc_1d a
		WHERE a.dt ='${outFileSuffix}' 
		AND a.cate_id IN (101,119) 
		AND a.highest_price>10 
		GROUP BY  a.token 
		         ,a.coupon_ratio 
		         ,a.qc_code 
		         ,a.highest_price 
	) a
	GROUP BY  a.token 
) d17
ON d17.token=d1.token
LEFT JOIN 
( --估价成功
	SELECT  a.token 
	       ,COUNT(distinct uid) AS num
	FROM hdp_ubu_zhuanzhuan_dw_c2b.dw_trade_recycle_eval_success_dtl_inc_1d a
	WHERE a.dt ='${outFileSuffix}' 
	AND a.cate_id IN (101,119) 
	GROUP BY  a.token 
) d18
ON d18.token=d1.token
WHERE d18.num=1 
AND d1.uid is not null 
AND d1.uid>0  