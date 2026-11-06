package com.contest.DriverGuard.terms.dao;

import com.contest.DriverGuard.terms.dto.TermsDto;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

import java.util.List;

@Mapper
public interface TermsAgreementDao {
    /** 현재 시행 중인 약관마다 이 사용자의 동의 여부 (기록이 없으면 agreed = false) */
    List<TermsDto.AgreementItem> findMyAgreements(@Param("userId") Long userId);

    /** 현재 필수 약관 중 동의하지 않은 것이 하나라도 있으면 true */
    boolean existsMissingRequired(@Param("userId") Long userId);

    /** 동의 기록 저장. 이미 동의한 약관에 다시 동의하면 처음 동의 시각을 유지 */
    void upsert(@Param("userId") Long userId, @Param("termsId") Long termsId, @Param("agreed") boolean agreed);

    /** 사용자의 동의 기록 전부 삭제 (탈퇴) */
    void deleteByUserId(@Param("userId") Long userId);
}
