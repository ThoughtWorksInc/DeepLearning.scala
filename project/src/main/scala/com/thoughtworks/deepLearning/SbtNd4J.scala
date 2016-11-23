package com.thoughtworks.deeplearning

import org.apache.commons.lang3.SystemUtils
import sbt.Keys._
import sbt._
import sbt.plugins.JvmPlugin

/**
  * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
  */
object SbtNd4J extends AutoPlugin {

  private def osClassifier(moduleId: ModuleID) = {
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

  def nd4jRuntime = osClassifier("org.nd4j" % "nd4j-native" % "0.4-rc3.9" classifier "")

  def addNd4jRuntime(configuration: Configuration) = Seq(
    classpathTypes += "maven-plugin",
    libraryDependencies += nd4jRuntime % Test
  )

  override def trigger = allRequirements

  override def requires = JvmPlugin

}
