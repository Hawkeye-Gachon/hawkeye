package com.contest.DriverGuard.terms.dao;

import com.contest.DriverGuard.terms.dto.TermsDto;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

import java.util.List;

@Mapper
public interface TermsDao {
    /** 현재 시행 중인 약관 (code 별 최신 버전), 표시 순서대로. content 제외 */
    List<TermsDto.Terms> findCurrent();

    /** 버전과 관계없이 ID 로 조회. 없으면 null */
    TermsDto.Terms findById(@Param("termsId") Long termsId);
}
