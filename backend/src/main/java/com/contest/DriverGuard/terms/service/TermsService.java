package com.contest.DriverGuard.terms.service;

import com.contest.DriverGuard.exception.ApiException;
import com.contest.DriverGuard.exception.ErrorCode;
import com.contest.DriverGuard.terms.dao.TermsAgreementDao;
import com.contest.DriverGuard.terms.dao.TermsDao;
import com.contest.DriverGuard.terms.dto.TermsDto;
import com.contest.DriverGuard.user.service.UserService;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class TermsService {

    private final TermsDao termsDao;
    private final TermsAgreementDao termsAgreementDao;
    private final UserService userService;

    @Transactional(readOnly = true)
    public TermsDto.ListResponse getCurrentList() {
        return new TermsDto.ListResponse(termsDao.findCurrent().stream().map(TermsDto.Summary::from).toList());
    }

    @Transactional(readOnly = true)
    public TermsDto.DetailResponse getDetail(Long termsId) {
        TermsDto.Terms terms = termsDao.findById(termsId);
        if (terms == null) {
            throw new ApiException(ErrorCode.TERMS_NOT_FOUND);
        }
        return TermsDto.DetailResponse.from(terms);
    }

    /** 현재 필수 약관 중 동의하지 않은 것이 있으면 true (신규 가입 미완료 또는 약관 개정) */
    @Transactional(readOnly = true)
    public boolean isAgreementRequired(Long userId) {
        return termsAgreementDao.existsMissingRequired(userId);
    }

    @Transactional(readOnly = true)
    public TermsDto.AgreementsResponse getMyAgreements(Long userId) {
        userService.getActiveUser(userId);   // 탈퇴한 사용자면 404
        List<TermsDto.AgreementItem> items = termsAgreementDao.findMyAgreements(userId);
        boolean required = items.stream().anyMatch(i -> i.getRequired() && !i.getAgreed());
        return new TermsDto.AgreementsResponse(required, items);
    }

    /**
     * 동의 저장. 보내지 않은 약관은 기존 상태 유지.
     * 저장 후 현재 필수 약관이 모두 동의 상태여야 함 (필수 약관을 false 로 보내거나, 동의한 적 없는 필수 약관을 빠뜨리면 400)
     */
    @Transactional
    public TermsDto.AgreementsResponse agree(Long userId, TermsDto.AgreeRequest request) {
        userService.getActiveUser(userId);   // 탈퇴한 사용자면 404

        // 같은 termsId 를 여러 번 보내면 마지막 값 사용
        Map<Long, Boolean> requested = new LinkedHashMap<>();
        request.getAgreements().forEach(a -> requested.put(a.getTermsId(), a.getAgreed()));

        List<TermsDto.AgreementItem> current = termsAgreementDao.findMyAgreements(userId);
        Set<Long> currentIds = current.stream().map(TermsDto.AgreementItem::getTermsId).collect(Collectors.toSet());

        for (Long termsId : requested.keySet()) {
            if (!currentIds.contains(termsId)) {
                throw new ApiException(termsDao.findById(termsId) == null
                        ? ErrorCode.TERMS_NOT_FOUND
                        : ErrorCode.TERMS_NOT_CURRENT);
            }
        }

        for (TermsDto.AgreementItem item : current) {
            if (!item.getRequired()) {
                continue;
            }
            Boolean agreed = requested.get(item.getTermsId());
            boolean agreedAfter = agreed != null ? agreed : item.getAgreed();
            if (!agreedAfter) {
                throw new ApiException(ErrorCode.TERMS_REQUIRED_NOT_AGREED);
            }
        }

        requested.forEach((termsId, agreed) -> termsAgreementDao.upsert(userId, termsId, agreed));
        return getMyAgreements(userId);
    }

    /** 탈퇴: 약관 동의 기록 전부 삭제 */
    @Transactional
    public void deleteAllByUser(Long userId) {
        termsAgreementDao.deleteByUserId(userId);
    }
}
