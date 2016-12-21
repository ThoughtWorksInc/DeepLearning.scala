sbt.dsl.dependsOn(BpBoolean, BpDouble, Bp2DArray, BpHList, BpCoproduct, BpSeq, BpAny, BpNothing)

lazy val Layer = project.disablePlugins(SparkPackagePlugin)

lazy val Lift = project.disablePlugins(SparkPackagePlugin).dependsOn(Layer)

lazy val BpBoolean = project.disablePlugins(SparkPackagePlugin).dependsOn(Layer, BufferedLayer, Poly)

lazy val BpDouble = project.disablePlugins(SparkPackagePlugin).dependsOn(Poly, BpBoolean, BufferedLayer)

lazy val Conversion = project.disablePlugins(SparkPackagePlugin).dependsOn(Layer)

lazy val Poly = project.disablePlugins(SparkPackagePlugin).dependsOn(Lift, Conversion)

lazy val BpAny = project.disablePlugins(SparkPackagePlugin).dependsOn(Lift, Conversion)

lazy val BpNothing = project.disablePlugins(SparkPackagePlugin).dependsOn(Lift, Conversion)

lazy val BpSeq = project.disablePlugins(SparkPackagePlugin).dependsOn(Lift, Conversion)

lazy val Bp2DArray = project.disablePlugins(SparkPackagePlugin).dependsOn(BpDouble)

lazy val BpHList = project.disablePlugins(SparkPackagePlugin).dependsOn(Poly)

lazy val BpCoproduct = project.disablePlugins(SparkPackagePlugin).dependsOn(BpBoolean)

lazy val BufferedLayer = project.disablePlugins(SparkPackagePlugin).dependsOn(Layer)

lazy val `sbt-nd4j` = project.disablePlugins(SparkPackagePlugin)

SbtNd4J.addNd4jRuntime(Test)

addCompilerPlugin("org.scalamacros" % "paradise" % "2.1.0" cross CrossVersion.full)

libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.0" % Test

crossScalaVersions := Seq("2.10.6", "2.11.8")

publishArtifact := false

lazy val unidoc = project
  .enablePlugins(TravisUnidocTitle)
  .settings(addCompilerPlugin("org.scalamacros" % "paradise" % "2.1.0" cross CrossVersion.full))

organization in ThisBuild := "com.thoughtworks.deeplearning"
