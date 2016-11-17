lazy val all = (project in file(".")).dependsOn(any, boolean, double, array2D, hlist, coproduct)

lazy val `deep-learning` = project.disablePlugins(SparkPackagePlugin)

lazy val boolean = project.disablePlugins(SparkPackagePlugin).dependsOn(any)

lazy val double = project.disablePlugins(SparkPackagePlugin).dependsOn(any, boolean)

lazy val any = project.disablePlugins(SparkPackagePlugin).dependsOn(`deep-learning`, syntax)

lazy val array2D = project.disablePlugins(SparkPackagePlugin).dependsOn(double)

lazy val hlist = project.disablePlugins(SparkPackagePlugin).dependsOn(any)

lazy val coproduct = project.disablePlugins(SparkPackagePlugin).dependsOn(boolean)

lazy val `sbt-nd4j` = project

lazy val syntax = project

import org.apache.commons.lang3.SystemUtils

libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.0" % Test

classpathTypes += "maven-plugin"

libraryDependencies += "org.nd4j" %% "nd4s" % "0.4-rc3.8"

libraryDependencies += "org.nd4j" % "nd4j-api" % "0.4-rc3.9"

def osClassifier(moduleId: ModuleID) = {
  val arch = SystemUtils.OS_ARCH match {
    case "amd64" => "x86_64"
    case other => other
  }
  if (SystemUtils.IS_OS_MAC_OSX) {
    moduleId classifier s"macosx-${arch}"
  } else if (SystemUtils.IS_OS_LINUX) {
    moduleId classifier s"linux-${arch}"
  } else if (SystemUtils.IS_OS_WINDOWS) {
    moduleId classifier s"windows-${arch}"
  } else {
    moduleId
  }
}

libraryDependencies += osClassifier("org.nd4j" % "nd4j-native" % "0.4-rc3.9" % Test classifier "")