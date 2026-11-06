package com.personal.driverguard.views.container

import androidx.compose.runtime.Composable
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.personal.driverguard.views.calibration.CalibrationScreen
import com.personal.driverguard.views.devsettings.DevSettingsScreen
import com.personal.driverguard.views.driving.DrivingScreen
import com.personal.driverguard.views.home.HomeScreen
import com.personal.driverguard.views.tripdetail.TripDetailScreen
import com.personal.driverguard.views.trips.TripListScreen

@Composable
fun AppNavHost() {
    val navController = rememberNavController()

    NavHost(navController = navController, startDestination = HomeRoute) {
        composable<HomeRoute> {
            HomeScreen(
                onCalibrate = { navController.navigate(CalibrationRoute) },
                onOpenTrips = { navController.navigate(TripListRoute) },
                onOpenDevSettings = { navController.navigate(DevSettingsRoute) },
            )
        }
        composable<CalibrationRoute> {
            CalibrationScreen(
                onBack = { navController.navigateUp() },
                onStartDriving = {
                    navController.navigate(DrivingRoute) {
                        popUpTo<CalibrationRoute> { inclusive = true }
                    }
                },
            )
        }
        composable<DrivingRoute> {
            DrivingScreen(
                onFinished = { tripId ->
                    // 주행 화면을 스택에서 빼고 방금 끝난 여정 상세를 보여 준다
                    navController.navigate(TripDetailRoute(tripId)) {
                        popUpTo<DrivingRoute> { inclusive = true }
                    }
                },
                onExit = { navController.navigateUp() },
            )
        }
        composable<TripListRoute> {
            TripListScreen(
                onBack = { navController.navigateUp() },
                onOpenTrip = { navController.navigate(TripDetailRoute(it)) },
            )
        }
        composable<TripDetailRoute> {
            TripDetailScreen(onBack = { navController.navigateUp() })
        }
        composable<DevSettingsRoute> {
            DevSettingsScreen(onBack = { navController.navigateUp() })
        }
    }
}
